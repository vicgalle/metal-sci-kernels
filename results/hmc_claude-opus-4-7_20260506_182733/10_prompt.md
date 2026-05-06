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
// Templated worker: D is compile-time. Uses float4 loads from threadgroup A.
// Avoids storing q_old (re-reads q_in on reject) to reduce register pressure.
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

    float4 q4[D4];
    float4 p4[D4];
    float4 f4[D4];

    uint base_counter = hmc_step_idx * (D + 1u);
    float eps_half = 0.5f * eps;

    // 1) Momentum p ~ N(0, I) via Box-Muller, paired counters.
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
        // pack into float4 lanes
        uint vi = i >> 2;
        uint li = i & 3u;
        if (li == 0u) { p4[vi].x = r * c; p4[vi].y = r * s; }
        else          { p4[vi].z = r * c; p4[vi].w = r * s; }
    }

    // 2) Load q (and remember source pointer for reject path).
    device const float4 *qin4 = (device const float4 *)(q_in + chain_idx * D);
    #pragma unroll
    for (uint v = 0u; v < D4; ++v) q4[v] = qin4[v];

    // matvec f = A q  (row-by-row, vectorized inner product)
    #pragma unroll
    for (uint i = 0u; i < D; ++i) {
        threadgroup const float4 *Arow4 = Atile4 + i * D4;
        float4 acc = float4(0.0f);
        #pragma unroll
        for (uint v = 0u; v < D4; ++v) {
            acc = fma(Arow4[v], q4[v], acc);
        }
        float a = acc.x + acc.y + acc.z + acc.w;
        uint vi = i >> 2;
        uint li = i & 3u;
        if      (li == 0u) f4[vi].x = a;
        else if (li == 1u) f4[vi].y = a;
        else if (li == 2u) f4[vi].z = a;
        else               f4[vi].w = a;
    }

    // U_old, K_old
    float4 Uacc = float4(0.0f);
    float4 Kacc = float4(0.0f);
    #pragma unroll
    for (uint v = 0u; v < D4; ++v) {
        Uacc = fma(q4[v], f4[v], Uacc);
        Kacc = fma(p4[v], p4[v], Kacc);
    }
    float U_old = 0.5f * (Uacc.x + Uacc.y + Uacc.z + Uacc.w);
    float K_old = 0.5f * (Kacc.x + Kacc.y + Kacc.z + Kacc.w);

    // 3) Leapfrog: initial half-kick
    #pragma unroll
    for (uint v = 0u; v < D4; ++v) p4[v] = fma(float4(-eps_half), f4[v], p4[v]);

    for (uint l = 0u; l < L; ++l) {
        // drift
        #pragma unroll
        for (uint v = 0u; v < D4; ++v) q4[v] = fma(float4(eps), p4[v], q4[v]);
        // recompute force
        #pragma unroll
        for (uint i = 0u; i < D; ++i) {
            threadgroup const float4 *Arow4 = Atile4 + i * D4;
            float4 acc = float4(0.0f);
            #pragma unroll
            for (uint v = 0u; v < D4; ++v) {
                acc = fma(Arow4[v], q4[v], acc);
            }
            float a = acc.x + acc.y + acc.z + acc.w;
            uint vi = i >> 2;
            uint li = i & 3u;
            if      (li == 0u) f4[vi].x = a;
            else if (li == 1u) f4[vi].y = a;
            else if (li == 2u) f4[vi].z = a;
            else               f4[vi].w = a;
        }
        // kick
        float scale = (l + 1u == L) ? -eps_half : -eps;
        #pragma unroll
        for (uint v = 0u; v < D4; ++v) p4[v] = fma(float4(scale), f4[v], p4[v]);
    }

    // 4) U_new, K_new (reuse final force)
    float4 Uacc2 = float4(0.0f);
    float4 Kacc2 = float4(0.0f);
    #pragma unroll
    for (uint v = 0u; v < D4; ++v) {
        Uacc2 = fma(q4[v], f4[v], Uacc2);
        Kacc2 = fma(p4[v], p4[v], Kacc2);
    }
    float U_new = 0.5f * (Uacc2.x + Uacc2.y + Uacc2.z + Uacc2.w);
    float K_new = 0.5f * (Kacc2.x + Kacc2.y + Kacc2.z + Kacc2.w);

    // 5) Accept/reject
    float dH = (U_new + K_new) - (U_old + K_old);
    uint b_acc = rand_u32(seed, chain_idx, base_counter + D);
    float u_acc = u01_from_bits(b_acc);
    float log_u = log(max(u_acc, 1.0e-30f));
    bool accept = isfinite(dH) && (log_u < -dH);

    device float4 *qout4 = (device float4 *)(q_out + chain_idx * D);
    if (accept) {
        #pragma unroll
        for (uint v = 0u; v < D4; ++v) qout4[v] = q4[v];
        accept_cnt[chain_idx] = accept_cnt[chain_idx] + 1u;
    } else {
        // Re-read q_in to write q_old (saves D registers vs caching qold).
        #pragma unroll
        for (uint v = 0u; v < D4; ++v) qout4[v] = qin4[v];
    }
}

// ============================================================================
// Entry kernel: cooperatively cache A into threadgroup as float4, dispatch by d.
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
         d8_K16384: correct, 5.49 ms, 787.4 GFLOPS (mat-vec FMAs only) (17.5% of 4500 GFLOPS)
         d16_K4096: correct, 14.61 ms, 306.0 GFLOPS (mat-vec FMAs only) (6.8% of 4500 GFLOPS)
         d32_K1024: correct, 34.47 ms, 131.8 GFLOPS (mat-vec FMAs only) (2.9% of 4500 GFLOPS)
  score (gmean of fraction): 0.0704

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

- iter  2: compile=OK | correct=True | score=0.018287045685874234
- iter  3: compile=OK | correct=True | score=0.0034231485888168525
- iter  4: compile=OK | correct=True | score=0.017603216634095798
- iter  5: compile=OK | correct=True | score=0.027114045324608524
- iter  6: compile=OK | correct=True | score=0.09316451457762556
- iter  7: compile=OK | correct=False | score=N/A
- iter  8: compile=OK | correct=True | score=0.09093050813467153
- iter  9: compile=OK | correct=True | score=0.0703651146705733

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
