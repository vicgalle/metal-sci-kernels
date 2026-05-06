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

## Your previous attempt

```metal
#include <metal_stdlib>
using namespace metal;

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
    return float(bits >> 8) * (1.0f / 16777216.0f);
}

// ============================================================================
// Templated worker: D is compile-time; uses float4 loads from threadgroup A.
// ============================================================================
template <uint D>
inline void hmc_run(uint chain_idx,
                    device const float *q_in,
                    device       float *q_out,
                    device       uint  *accept_cnt,
                    threadgroup const float4 *Atile4, // row stride = D/4 float4s
                    uint L, float eps,
                    uint hmc_step_idx, uint seed)
{
    constexpr uint D4 = D / 4u;

    float q[D];
    float p[D];
    float f[D];

    uint base_counter = hmc_step_idx * (D + 1u);
    float eps_half = 0.5f * eps;

    // 1) Momentum p ~ N(0, I)
    #pragma unroll
    for (uint i = 0u; i < D; i += 2u) {
        uint b1 = rand_u32(seed, chain_idx, base_counter + i);
        uint b2 = rand_u32(seed, chain_idx, base_counter + i + 1u);
        float u1 = u01_from_bits(b1);
        float u2 = u01_from_bits(b2);
        u1 = max(u1, 1.0e-7f);
        float r = sqrt(-2.0f * log(u1));
        float angle = 6.2831853071795864f * u2;
        float c;
        float s = sincos(angle, c);
        p[i] = r * c;
        p[i + 1u] = r * s;
    }

    // 2) Load q
    device const float *qin = q_in + chain_idx * D;
    #pragma unroll
    for (uint i = 0u; i < D; ++i) q[i] = qin[i];

    // matvec f = A q with float4 loads from threadgroup
    #pragma unroll
    for (uint i = 0u; i < D; ++i) {
        threadgroup const float4 *Arow4 = Atile4 + i * D4;
        float4 acc4 = float4(0.0f);
        #pragma unroll
        for (uint j4 = 0u; j4 < D4; ++j4) {
            float4 a = Arow4[j4];
            float4 qv = float4(q[j4*4u], q[j4*4u+1u], q[j4*4u+2u], q[j4*4u+3u]);
            acc4 = fma(a, qv, acc4);
        }
        f[i] = acc4.x + acc4.y + acc4.z + acc4.w;
    }

    // U_old, K_old
    float U_old = 0.0f;
    float K_old = 0.0f;
    #pragma unroll
    for (uint i = 0u; i < D; ++i) {
        U_old = fma(q[i], f[i], U_old);
        K_old = fma(p[i], p[i], K_old);
    }
    U_old *= 0.5f;
    K_old *= 0.5f;

    // 3) Leapfrog
    // initial half-kick
    #pragma unroll
    for (uint i = 0u; i < D; ++i) p[i] = fma(-eps_half, f[i], p[i]);

    for (uint l = 0u; l < L; ++l) {
        // drift
        #pragma unroll
        for (uint i = 0u; i < D; ++i) q[i] = fma(eps, p[i], q[i]);
        // recompute force with float4 loads
        #pragma unroll
        for (uint i = 0u; i < D; ++i) {
            threadgroup const float4 *Arow4 = Atile4 + i * D4;
            float4 acc4 = float4(0.0f);
            #pragma unroll
            for (uint j4 = 0u; j4 < D4; ++j4) {
                float4 a = Arow4[j4];
                float4 qv = float4(q[j4*4u], q[j4*4u+1u], q[j4*4u+2u], q[j4*4u+3u]);
                acc4 = fma(a, qv, acc4);
            }
            f[i] = acc4.x + acc4.y + acc4.z + acc4.w;
        }
        // kick
        float scale = (l + 1u == L) ? -eps_half : -eps;
        #pragma unroll
        for (uint i = 0u; i < D; ++i) p[i] = fma(scale, f[i], p[i]);
    }

    // 4) U_new, K_new
    float U_new = 0.0f;
    float K_new = 0.0f;
    #pragma unroll
    for (uint i = 0u; i < D; ++i) {
        U_new = fma(q[i], f[i], U_new);
        K_new = fma(p[i], p[i], K_new);
    }
    U_new *= 0.5f;
    K_new *= 0.5f;

    // 5) Accept/reject
    float dH = (U_new + K_new) - (U_old + K_old);
    uint b_acc = rand_u32(seed, chain_idx, base_counter + D);
    float u_acc = u01_from_bits(b_acc);
    float log_u = log(max(u_acc, 1.0e-30f));
    bool accept = isfinite(dH) && (log_u < -dH);

    device float *qout = q_out + chain_idx * D;
    if (accept) {
        #pragma unroll
        for (uint i = 0u; i < D; ++i) qout[i] = q[i];
        accept_cnt[chain_idx] = accept_cnt[chain_idx] + 1u;
    } else {
        // Reject: write q_old (== q_in)
        #pragma unroll
        for (uint i = 0u; i < D; ++i) qout[i] = qin[i];
    }
}

// ============================================================================
// Entry: cooperatively cache A into threadgroup as float4, then dispatch by d.
// ============================================================================
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
                     uint chain_idx [[thread_position_in_grid]],
                     uint tid_in_tg [[thread_position_in_threadgroup]],
                     uint tg_size   [[threads_per_threadgroup]])
{
    // 32x32 = 1024 floats = 256 float4 = 4 KB.
    threadgroup float4 Atile4[256];
    threadgroup float *Atile = (threadgroup float *)Atile4;

    uint dd = d * d;
    for (uint k = tid_in_tg; k < dd; k += tg_size) {
        Atile[k] = A[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (chain_idx >= K) return;

    if (d == 8u) {
        hmc_run<8u>(chain_idx, q_in, q_out, accept_cnt, Atile4, L, eps, hmc_step_idx, seed);
    } else if (d == 16u) {
        hmc_run<16u>(chain_idx, q_in, q_out, accept_cnt, Atile4, L, eps, hmc_step_idx, seed);
    } else {
        hmc_run<32u>(chain_idx, q_in, q_out, accept_cnt, Atile4, L, eps, hmc_step_idx, seed);
    }
}
```

Result of previous attempt:
         d8_K16384: correct, 5.49 ms, 787.7 GFLOPS (mat-vec FMAs only) (17.5% of 4500 GFLOPS)
         d16_K4096: correct, 8.87 ms, 504.0 GFLOPS (mat-vec FMAs only) (11.2% of 4500 GFLOPS)
         d32_K1024: correct, 26.31 ms, 172.6 GFLOPS (mat-vec FMAs only) (3.8% of 4500 GFLOPS)
  score (gmean of fraction): 0.0909

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

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
    return float(bits >> 8) * (1.0f / 16777216.0f);
}

// ============================================================================
// Templated worker: D is a compile-time size; matvec/reductions fully unroll.
// ============================================================================
template <uint D>
inline void hmc_run(uint chain_idx,
                    device const float *q_in,
                    device       float *q_out,
                    device       uint  *accept_cnt,
                    threadgroup const float *Atile,   // row stride = D
                    uint L, float eps,
                    uint hmc_step_idx, uint seed)
{
    float q[D];
    float p[D];
    float f[D];
    float qold[D];

    uint base_counter = hmc_step_idx * (D + 1u);
    float eps_half = 0.5f * eps;

    // 1) Momentum p ~ N(0, I)
    #pragma unroll
    for (uint i = 0u; i < D; i += 2u) {
        uint b1 = rand_u32(seed, chain_idx, base_counter + i);
        uint b2 = rand_u32(seed, chain_idx, base_counter + i + 1u);
        float u1 = u01_from_bits(b1);
        float u2 = u01_from_bits(b2);
        u1 = max(u1, 1.0e-7f);
        float r = sqrt(-2.0f * log(u1));
        float angle = 6.2831853071795864f * u2;
        float c;
        float s = sincos(angle, c);
        p[i] = r * c;
        p[i + 1u] = r * s;
    }

    // 2) Load q, save q_old
    device const float *qin = q_in + chain_idx * D;
    #pragma unroll
    for (uint i = 0u; i < D; ++i) {
        float qi = qin[i];
        q[i] = qi;
        qold[i] = qi;
    }

    // matvec f = A q  (fully unrolled)
    #pragma unroll
    for (uint i = 0u; i < D; ++i) {
        threadgroup const float *Arow = Atile + i * D;
        float acc = 0.0f;
        #pragma unroll
        for (uint j = 0u; j < D; ++j) {
            acc = fma(Arow[j], q[j], acc);
        }
        f[i] = acc;
    }

    // U_old, K_old
    float U_old = 0.0f;
    float K_old = 0.0f;
    #pragma unroll
    for (uint i = 0u; i < D; ++i) {
        U_old = fma(q[i], f[i], U_old);
        K_old = fma(p[i], p[i], K_old);
    }
    U_old *= 0.5f;
    K_old *= 0.5f;

    // 3) Leapfrog
    // Initial half-kick
    #pragma unroll
    for (uint i = 0u; i < D; ++i) p[i] = fma(-eps_half, f[i], p[i]);

    for (uint l = 0u; l < L; ++l) {
        // drift
        #pragma unroll
        for (uint i = 0u; i < D; ++i) q[i] = fma(eps, p[i], q[i]);
        // recompute force
        #pragma unroll
        for (uint i = 0u; i < D; ++i) {
            threadgroup const float *Arow = Atile + i * D;
            float acc = 0.0f;
            #pragma unroll
            for (uint j = 0u; j < D; ++j) {
                acc = fma(Arow[j], q[j], acc);
            }
            f[i] = acc;
        }
        // kick
        float scale = (l + 1u == L) ? -eps_half : -eps;
        #pragma unroll
        for (uint i = 0u; i < D; ++i) p[i] = fma(scale, f[i], p[i]);
    }

    // 4) U_new, K_new
    float U_new = 0.0f;
    float K_new = 0.0f;
    #pragma unroll
    for (uint i = 0u; i < D; ++i) {
        U_new = fma(q[i], f[i], U_new);
        K_new = fma(p[i], p[i], K_new);
    }
    U_new *= 0.5f;
    K_new *= 0.5f;

    // 5) Accept/reject
    float dH = (U_new + K_new) - (U_old + K_old);
    uint b_acc = rand_u32(seed, chain_idx, base_counter + D);
    float u_acc = u01_from_bits(b_acc);
    float log_u = log(max(u_acc, 1.0e-30f));
    bool accept = isfinite(dH) && (log_u < -dH);

    device float *qout = q_out + chain_idx * D;
    #pragma unroll
    for (uint i = 0u; i < D; ++i) {
        qout[i] = accept ? q[i] : qold[i];
    }
    if (accept) {
        accept_cnt[chain_idx] = accept_cnt[chain_idx] + 1u;
    }
}

// ============================================================================
// Entry kernel: cooperatively cache A into threadgroup, then dispatch by d.
// ============================================================================
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
                     uint chain_idx [[thread_position_in_grid]],
                     uint tid_in_tg [[thread_position_in_threadgroup]],
                     uint tg_size   [[threads_per_threadgroup]])
{
    // Tightly-packed A cache: d * d floats (max 32*32 = 4 KB).
    threadgroup float Atile[32u * 32u];

    uint dd = d * d;
    for (uint k = tid_in_tg; k < dd; k += tg_size) {
        Atile[k] = A[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (chain_idx >= K) return;

    if (d == 8u) {
        hmc_run<8u>(chain_idx, q_in, q_out, accept_cnt, Atile, L, eps, hmc_step_idx, seed);
    } else if (d == 16u) {
        hmc_run<16u>(chain_idx, q_in, q_out, accept_cnt, Atile, L, eps, hmc_step_idx, seed);
    } else {
        hmc_run<32u>(chain_idx, q_in, q_out, accept_cnt, Atile, L, eps, hmc_step_idx, seed);
    }
}
```

Incumbent result:
         d8_K16384: correct, 4.46 ms, 969.7 GFLOPS (mat-vec FMAs only) (21.5% of 4500 GFLOPS)
         d16_K4096: correct, 8.11 ms, 551.0 GFLOPS (mat-vec FMAs only) (12.2% of 4500 GFLOPS)
         d32_K1024: correct, 32.93 ms, 137.9 GFLOPS (mat-vec FMAs only) (3.1% of 4500 GFLOPS)
  score (gmean of fraction): 0.0932

## History

- iter  1: compile=OK | correct=True | score=0.008896577055895305
- iter  2: compile=OK | correct=True | score=0.018287045685874234
- iter  3: compile=OK | correct=True | score=0.0034231485888168525
- iter  4: compile=OK | correct=True | score=0.017603216634095798
- iter  5: compile=OK | correct=True | score=0.027114045324608524
- iter  6: compile=OK | correct=True | score=0.09316451457762556
- iter  7: compile=OK | correct=False | score=N/A
- iter  8: compile=OK | correct=True | score=0.09093050813467153

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
