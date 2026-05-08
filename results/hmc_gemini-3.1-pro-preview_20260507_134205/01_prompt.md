## Task: hmc

Hamiltonian Monte Carlo on a multivariate Gaussian target with mean 0 and precision matrix A = Sigma^{-1} (provided as a (d, d) row-major float32 buffer). One thread per chain; many chains run in parallel.

Per HMC step (one dispatch):
  1) p ~ N(0, I): for each pair (i, i+1) in 0..d-1 step 2,
     draw two uniforms u1, u2 in [0, 1) via the prescribed RNG
     (counters base_counter + i and base_counter + i + 1, where
     base_counter = hmc_step_idx * (d + 1)) and apply Box-Muller:
        u1 = max(u1, 1e-7);  r = sqrt(-2 * log(u1));
        angle = 2 pi * u2;
        p[i]   = r * cos(angle);
        p[i+1] = r * sin(angle);   // skip if i+1 >= d (d is even).
  2) Save q_old = q. Compute force = A q;
     U_old = 0.5 * dot(q, force); K_old = 0.5 * dot(p, p).
  3) Leapfrog with eps:
        p   -= (eps/2) * force            // initial half-kick
        for l = 0..L-1:
            q   += eps * p                // drift
            force = A q                   // recompute force at new q
            scale = (l + 1 == L) ? (eps/2) : eps
            p   -= scale * force          // kick
  4) U_new = 0.5 * dot(q, force) [reusing the final force];
     K_new = 0.5 * dot(p, p);
     dH = (U_new + K_new) - (U_old + K_old).
  5) Draw uniform u_acc with counter base_counter + d.
     accept = isfinite(dH) AND log(max(u_acc, 1e-30)) < -dH.
     Write q if accept else q_old to q_out[chain_idx * d + i];
     if accept, accept_cnt[chain_idx] += 1.

RNG (must be reproduced bit-exactly):
  inline uint mix32(uint x) {
      x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
      x = (x ^ (x >> 13)) * 0xC2B2AE35u;
      return x ^ (x >> 16);
  }
  uint x = seed + chain_idx * 0x9E3779B9u;
  x = mix32(x ^ counter);
  x = mix32(x);
  float u = float(x >> 8) * (1.0f / 16777216.0f);

The host ping-pongs (q_in, q_out) buffers across HMC steps; all dispatches share one command buffer for end-to-end timing.

## Required kernel signature(s)

```
kernel void hmc_step(device const float *q_in        [[buffer(0)]],
                     device       float *q_out       [[buffer(1)]],
                     device       uint  *accept_cnt  [[buffer(2)]],
                     device const float *A           [[buffer(3)]],
                     constant uint  &K               [[buffer(4)]],
                     constant uint  &d               [[buffer(5)]],
                     constant uint  &L               [[buffer(6)]],
                     constant float &eps             [[buffer(7)]],
                     constant uint  &hmc_step_idx    [[buffer(8)]],
                     constant uint  &seed            [[buffer(9)]],
                     uint chain_idx [[thread_position_in_grid]]);

Threads are dispatched 1-D, one per chain; guard with `if (chain_idx >= K) return;`. The host ping-pongs (q_in, q_out) between two K * d float buffers and increments hmc_step_idx by 1 per dispatch. accept_cnt is initialised to zero and accumulates accepted proposals over the run.

All chosen sizes satisfy d <= 32 and d is even; thread-private arrays of size 32 are sufficient. Threadgroup-cooperative schemes (multiple threads per chain sharing the mat-vec) and simdgroup reductions are valid optimisations as long as the external buffer layout above is preserved.
```

## Baseline: naive seed kernel

```metal
// Naive seed kernel for Hamiltonian Monte Carlo on a Gaussian target.
// One thread per chain.
//
// Target: U(q) = (1/2) q^T A q  with A = Sigma^{-1} symmetric PD.
// Per HMC step (one dispatch):
//   1. Sample momentum p ~ N(0, I) via Box-Muller (d Gaussians per chain).
//   2. Save q_old. Compute U_old, K_old.
//   3. Leapfrog with step size eps for L steps:
//        p   <- p - (eps/2) A q                  // initial half-kick
//        for l = 1..L-1:
//            q <- q + eps p                       // drift
//            p <- p - eps A q                     // full kick
//        q   <- q + eps p                         // last drift
//        p   <- p - (eps/2) A q                   // final half-kick
//   4. Compute U_new, K_new. dH = (U_new + K_new) - (U_old + K_old).
//   5. Accept iff log(u_acc) < -dH (equivalently u_acc < exp(-dH));
//      otherwise rollback q <- q_old. Increment per-chain accept count
//      on accept.
//
// RNG: Murmur3-fmix32 hash of (seed, chain_idx, base_counter + sample_idx)
// where base_counter = hmc_step_idx * (d + 1). Per HMC step per chain we
// need d uniforms for the momentum (paired into d/2 Box-Muller calls)
// plus 1 uniform for accept.
//
// Per-thread storage uses thread-private arrays sized to a compile-time
// upper bound D_MAX. The host's sizes all satisfy d <= D_MAX = 32.
//
// Buffer layout (must be preserved by candidate kernels):
//   buffer 0: device const float *q_in        (K * d)
//   buffer 1: device       float *q_out       (K * d)
//   buffer 2: device       uint  *accept_cnt  (K, accumulates over HMC steps)
//   buffer 3: device const float *A           (d * d, row-major precision matrix)
//   buffer 4: const uint  &K
//   buffer 5: const uint  &d
//   buffer 6: const uint  &L
//   buffer 7: const float &eps
//   buffer 8: const uint  &hmc_step_idx
//   buffer 9: const uint  &seed

#include <metal_stdlib>
using namespace metal;

#define D_MAX 32u

inline uint mix32(uint x) {
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    x = x ^ (x >> 16);
    return x;
}

inline uint rand_u32(uint seed, uint chain_idx, uint counter) {
    uint x = seed + chain_idx * 0x9E3779B9u;
    x = mix32(x ^ counter);
    x = mix32(x);
    return x;
}

inline float u01_from_bits(uint bits) {
    // 24-bit uniform in [0, 1); exact in fp32 on every platform.
    return float(bits >> 8) * (1.0f / 16777216.0f);
}

kernel void hmc_step(device const float *q_in        [[buffer(0)]],
                     device       float *q_out       [[buffer(1)]],
                     device       uint  *accept_cnt  [[buffer(2)]],
                     device const float *A           [[buffer(3)]],
                     constant uint  &K               [[buffer(4)]],
                     constant uint  &d               [[buffer(5)]],
                     constant uint  &L               [[buffer(6)]],
                     constant float &eps             [[buffer(7)]],
                     constant uint  &hmc_step_idx    [[buffer(8)]],
                     constant uint  &seed            [[buffer(9)]],
                     uint chain_idx [[thread_position_in_grid]]) {
    if (chain_idx >= K) return;

    thread float q[D_MAX];
    thread float p[D_MAX];
    thread float q_old[D_MAX];
    thread float force[D_MAX];

    uint base_counter = hmc_step_idx * (d + 1u);

    // 1. Momentum p ~ N(0, I), Box-Muller over (u1, u2) pairs.
    for (uint i = 0u; i < d; i += 2u) {
        uint b1 = rand_u32(seed, chain_idx, base_counter + i);
        uint b2 = rand_u32(seed, chain_idx, base_counter + i + 1u);
        float u1 = u01_from_bits(b1);
        float u2 = u01_from_bits(b2);
        u1 = max(u1, 1.0e-7f);                           // guard log(0)
        float r = sqrt(-2.0f * log(u1));
        float angle = 6.2831853071795864f * u2;          // 2 pi
        float c, s;
        s = sincos(angle, c);
        p[i] = r * c;
        if (i + 1u < d) {
            p[i + 1u] = r * s;
        }
    }

    // 2. Load q from q_in, save q_old.
    for (uint i = 0u; i < d; ++i) {
        float qi = q_in[chain_idx * d + i];
        q[i] = qi;
        q_old[i] = qi;
    }

    // Helper: force[i] = sum_j A[i, j] * q[j]; folds U = (1/2) q . force.
    float U_old = 0.0f;
    for (uint i = 0u; i < d; ++i) {
        float acc = 0.0f;
        for (uint j = 0u; j < d; ++j) {
            acc += A[i * d + j] * q[j];
        }
        force[i] = acc;
        U_old += 0.5f * q[i] * acc;
    }
    float K_old = 0.0f;
    for (uint i = 0u; i < d; ++i) K_old += 0.5f * p[i] * p[i];

    // 3. Leapfrog. We already have force = A q at q_0, so use it for
    //    the initial half-kick without a redundant mat-vec.
    for (uint i = 0u; i < d; ++i) p[i] -= 0.5f * eps * force[i];

    for (uint l = 0u; l < L; ++l) {
        // drift
        for (uint i = 0u; i < d; ++i) q[i] += eps * p[i];
        // recompute force at new q
        for (uint i = 0u; i < d; ++i) {
            float acc = 0.0f;
            for (uint j = 0u; j < d; ++j) acc += A[i * d + j] * q[j];
            force[i] = acc;
        }
        // kick (full step except after the last drift, where it's a half)
        float scale = (l + 1u == L) ? (0.5f * eps) : eps;
        for (uint i = 0u; i < d; ++i) p[i] -= scale * force[i];
    }

    // 4. New Hamiltonian. We have force = A q at the final q.
    float U_new = 0.0f;
    for (uint i = 0u; i < d; ++i) U_new += 0.5f * q[i] * force[i];
    float K_new = 0.0f;
    for (uint i = 0u; i < d; ++i) K_new += 0.5f * p[i] * p[i];

    // 5. Accept iff log(u_acc) < -dH (handles dH < 0 and dH > 0
    // uniformly; avoids exp() overflow on rejected proposals).
    float dH = (U_new + K_new) - (U_old + K_old);
    uint b_acc = rand_u32(seed, chain_idx, base_counter + d);
    float u_acc = u01_from_bits(b_acc);
    float log_u = log(max(u_acc, 1.0e-30f));
    bool accept = isfinite(dH) && (log_u < -dH);

    for (uint i = 0u; i < d; ++i) {
        q_out[chain_idx * d + i] = accept ? q[i] : q_old[i];
    }
    if (accept) {
        accept_cnt[chain_idx] = accept_cnt[chain_idx] + 1u;
    }
}
```

Measured baseline (seed):
         d8_K16384: correct, 48.77 ms, 88.7 GFLOPS (mat-vec FMAs only) (2.0% of 4500 GFLOPS)
         d16_K4096: correct, 109.34 ms, 40.9 GFLOPS (mat-vec FMAs only) (0.9% of 4500 GFLOPS)
         d32_K1024: correct, 335.53 ms, 13.5 GFLOPS (mat-vec FMAs only) (0.3% of 4500 GFLOPS)
  score (gmean of fraction): 0.0081

## Your task

Write an improved Metal kernel that produces correct results AND runs
faster than the seed across all problem sizes. The fitness score is
the geometric mean of `achieved / ceiling` across sizes; score 0 if
any size fails correctness.

Output ONE fenced ```metal``` code block containing the kernel(s).
Preserve the kernel name(s) and buffer indices exactly.
