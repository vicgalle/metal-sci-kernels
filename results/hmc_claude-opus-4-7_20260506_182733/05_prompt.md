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

#define D_MAX 32u
#define D_MAX4 8u

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
                     uint tg_size   [[threads_per_threadgroup]]) {
    threadgroup float4 Atile4[D_MAX * D_MAX4]; // row stride D_MAX4 float4

    uint dpad = (d + 3u) & ~3u;
    uint dpad4 = dpad >> 2;

    // Cooperative load of A: zero pad, place row i into Atile4[i*D_MAX4 .. i*D_MAX4+dpad4-1].
    uint tile_floats = D_MAX * D_MAX;
    threadgroup float *Atile = (threadgroup float *)Atile4;
    for (uint k = tid_in_tg; k < tile_floats; k += tg_size) {
        Atile[k] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint dd = d * d;
    for (uint k = tid_in_tg; k < dd; k += tg_size) {
        uint i = k / d;
        uint j = k - i * d;
        Atile[i * (D_MAX4 * 4u) + j] = A[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (chain_idx >= K) return;

    // Register arrays of float4.
    float4 q4[D_MAX4];
    float4 p4[D_MAX4];
    float4 f4[D_MAX4];
    float4 qold4[D_MAX4];

    for (uint i = 0u; i < D_MAX4; ++i) {
        q4[i] = float4(0.0f);
        p4[i] = float4(0.0f);
        f4[i] = float4(0.0f);
        qold4[i] = float4(0.0f);
    }

    uint base_counter = hmc_step_idx * (d + 1u);
    float eps_half = 0.5f * eps;

    // 1. Momentum p ~ N(0, I).
    {
        thread float *p = (thread float *)p4;
        for (uint i = 0u; i < d; i += 2u) {
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
            if (i + 1u < d) p[i + 1u] = r * s;
        }
    }

    // 2. Load q.
    {
        thread float *q = (thread float *)q4;
        thread float *qo = (thread float *)qold4;
        device const float *qin = q_in + chain_idx * d;
        for (uint i = 0u; i < d; ++i) {
            float qi = qin[i];
            q[i] = qi;
            qo[i] = qi;
        }
    }

    // Initial force and U_old.
    float U_old = 0.0f;
    for (uint i = 0u; i < d; ++i) {
        threadgroup const float4 *Arow = Atile4 + i * D_MAX4;
        float4 acc4 = float4(0.0f);
        for (uint j4 = 0u; j4 < dpad4; ++j4) {
            acc4 = fma(Arow[j4], q4[j4], acc4);
        }
        float acc = acc4.x + acc4.y + acc4.z + acc4.w;
        ((thread float *)f4)[i] = acc;
        U_old = fma(0.5f * ((thread float *)q4)[i], acc, U_old);
    }

    float K_old = 0.0f;
    for (uint j4 = 0u; j4 < dpad4; ++j4) {
        K_old += dot(p4[j4], p4[j4]);
    }
    K_old *= 0.5f;

    // Initial half-kick: p -= (eps/2) * f.
    float4 nh = float4(-eps_half);
    for (uint j4 = 0u; j4 < dpad4; ++j4) {
        p4[j4] = fma(nh, f4[j4], p4[j4]);
    }

    // Leapfrog loop. Last iter uses scale=eps_half; others eps.
    float4 e4 = float4(eps);
    float4 ne4 = float4(-eps);
    float4 nh4 = float4(-eps_half);

    if (L > 0u) {
        // First L-1 iterations with scale = eps.
        for (uint l = 0u; l + 1u < L; ++l) {
            for (uint j4 = 0u; j4 < dpad4; ++j4) {
                q4[j4] = fma(e4, p4[j4], q4[j4]);
            }
            for (uint i = 0u; i < d; ++i) {
                threadgroup const float4 *Arow = Atile4 + i * D_MAX4;
                float4 acc4 = float4(0.0f);
                for (uint j4 = 0u; j4 < dpad4; ++j4) {
                    acc4 = fma(Arow[j4], q4[j4], acc4);
                }
                ((thread float *)f4)[i] = acc4.x + acc4.y + acc4.z + acc4.w;
            }
            for (uint j4 = 0u; j4 < dpad4; ++j4) {
                p4[j4] = fma(ne4, f4[j4], p4[j4]);
            }
        }
        // Final iteration: scale = eps_half.
        for (uint j4 = 0u; j4 < dpad4; ++j4) {
            q4[j4] = fma(e4, p4[j4], q4[j4]);
        }
        for (uint i = 0u; i < d; ++i) {
            threadgroup const float4 *Arow = Atile4 + i * D_MAX4;
            float4 acc4 = float4(0.0f);
            for (uint j4 = 0u; j4 < dpad4; ++j4) {
                acc4 = fma(Arow[j4], q4[j4], acc4);
            }
            ((thread float *)f4)[i] = acc4.x + acc4.y + acc4.z + acc4.w;
        }
        for (uint j4 = 0u; j4 < dpad4; ++j4) {
            p4[j4] = fma(nh4, f4[j4], p4[j4]);
        }
    }

    // U_new and K_new.
    float U_new = 0.0f;
    {
        thread float *q = (thread float *)q4;
        thread float *f = (thread float *)f4;
        for (uint i = 0u; i < d; ++i) U_new = fma(0.5f * q[i], f[i], U_new);
    }
    float K_new = 0.0f;
    for (uint j4 = 0u; j4 < dpad4; ++j4) {
        K_new += dot(p4[j4], p4[j4]);
    }
    K_new *= 0.5f;

    float dH = (U_new + K_new) - (U_old + K_old);
    uint b_acc = rand_u32(seed, chain_idx, base_counter + d);
    float u_acc = u01_from_bits(b_acc);
    float log_u = log(max(u_acc, 1.0e-30f));
    bool accept = isfinite(dH) && (log_u < -dH);

    {
        thread float *q = (thread float *)q4;
        thread float *qo = (thread float *)qold4;
        device float *qout = q_out + chain_idx * d;
        for (uint i = 0u; i < d; ++i) {
            qout[i] = accept ? q[i] : qo[i];
        }
    }
    if (accept) {
        accept_cnt[chain_idx] = accept_cnt[chain_idx] + 1u;
    }
}
```

Result of previous attempt:
         d8_K16384: correct, 39.65 ms, 109.1 GFLOPS (mat-vec FMAs only) (2.4% of 4500 GFLOPS)
         d16_K4096: correct, 44.58 ms, 100.3 GFLOPS (mat-vec FMAs only) (2.2% of 4500 GFLOPS)
         d32_K1024: correct, 99.93 ms, 45.4 GFLOPS (mat-vec FMAs only) (1.0% of 4500 GFLOPS)
  score (gmean of fraction): 0.0176

## Current best (incumbent)

```metal
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
                     uint chain_idx [[thread_position_in_grid]],
                     uint tid_in_tg [[thread_position_in_threadgroup]],
                     uint tg_size   [[threads_per_threadgroup]]) {
    // Padded row stride (multiple of 4) so we can load float4 chunks safely.
    // d <= 32, so dpad <= 32.
    threadgroup float Atile[D_MAX * D_MAX];

    uint dpad = (d + 3u) & ~3u;

    // Cooperative load: zero-fill the tile, then place A[i,j] at Atile[i*dpad+j].
    uint tile_sz = D_MAX * D_MAX;
    for (uint k = tid_in_tg; k < tile_sz; k += tg_size) {
        Atile[k] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint dd = d * d;
    for (uint k = tid_in_tg; k < dd; k += tg_size) {
        uint i = k / d;
        uint j = k - i * d;
        Atile[i * dpad + j] = A[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (chain_idx >= K) return;

    // Use float arrays sized to a multiple of 4 so float4 loads are valid.
    thread float qbuf[D_MAX];
    thread float pbuf[D_MAX];
    thread float qold[D_MAX];
    thread float fbuf[D_MAX];

    // Zero-init tail (indices >= d) so float4 ops see 0 in padded slots.
    for (uint i = 0u; i < D_MAX; ++i) { qbuf[i] = 0.0f; }

    uint base_counter = hmc_step_idx * (d + 1u);
    float eps_half = 0.5f * eps;
    uint dpad4 = dpad >> 2;

    // 1. Momentum p ~ N(0, I), Box-Muller (counters i, i+1 for pair starting at i).
    for (uint i = 0u; i < d; i += 2u) {
        uint b1 = rand_u32(seed, chain_idx, base_counter + i);
        uint b2 = rand_u32(seed, chain_idx, base_counter + i + 1u);
        float u1 = u01_from_bits(b1);
        float u2 = u01_from_bits(b2);
        u1 = max(u1, 1.0e-7f);
        float r = sqrt(-2.0f * log(u1));
        float angle = 6.2831853071795864f * u2;
        float c;
        float s = sincos(angle, c);
        pbuf[i] = r * c;
        if (i + 1u < d) {
            pbuf[i + 1u] = r * s;
        }
    }
    // Zero the padded tail of pbuf so dot products are unaffected.
    for (uint i = d; i < dpad; ++i) pbuf[i] = 0.0f;

    // 2. Load q, save q_old.
    for (uint i = 0u; i < d; ++i) {
        float qi = q_in[chain_idx * d + i];
        qbuf[i] = qi;
        qold[i] = qi;
    }
    // qbuf tail already 0 from init.

    // Helper: compute force = A * q via threadgroup-mem float4 loads.
    // Inlined below.

    // Initial force and U_old.
    float U_old = 0.0f;
    for (uint i = 0u; i < d; ++i) {
        threadgroup const float4 *Arow = (threadgroup const float4 *)(Atile + i * dpad);
        float4 acc4 = float4(0.0f);
        thread const float4 *qv = (thread const float4 *)qbuf;
        for (uint j4 = 0u; j4 < dpad4; ++j4) {
            acc4 = fma(Arow[j4], qv[j4], acc4);
        }
        float acc = acc4.x + acc4.y + acc4.z + acc4.w;
        fbuf[i] = acc;
        U_old = fma(0.5f * qbuf[i], acc, U_old);
    }
    // Pad fbuf tail.
    for (uint i = d; i < dpad; ++i) fbuf[i] = 0.0f;

    float K_old = 0.0f;
    for (uint i = 0u; i < d; ++i) K_old = fma(0.5f * pbuf[i], pbuf[i], K_old);

    // 3. Leapfrog.
    // Initial half-kick: p -= (eps/2) * force.
    {
        thread float4 *pv = (thread float4 *)pbuf;
        thread const float4 *fv = (thread const float4 *)fbuf;
        for (uint j4 = 0u; j4 < dpad4; ++j4) {
            pv[j4] = fma(float4(-eps_half), fv[j4], pv[j4]);
        }
    }

    for (uint l = 0u; l < L; ++l) {
        // Drift: q += eps * p
        {
            thread float4 *qv = (thread float4 *)qbuf;
            thread const float4 *pv = (thread const float4 *)pbuf;
            for (uint j4 = 0u; j4 < dpad4; ++j4) {
                qv[j4] = fma(float4(eps), pv[j4], qv[j4]);
            }
        }
        // Recompute force = A q.
        for (uint i = 0u; i < d; ++i) {
            threadgroup const float4 *Arow = (threadgroup const float4 *)(Atile + i * dpad);
            float4 acc4 = float4(0.0f);
            thread const float4 *qv = (thread const float4 *)qbuf;
            for (uint j4 = 0u; j4 < dpad4; ++j4) {
                acc4 = fma(Arow[j4], qv[j4], acc4);
            }
            fbuf[i] = acc4.x + acc4.y + acc4.z + acc4.w;
        }
        float scale = (l + 1u == L) ? eps_half : eps;
        {
            thread float4 *pv = (thread float4 *)pbuf;
            thread const float4 *fv = (thread const float4 *)fbuf;
            float4 sc4 = float4(-scale);
            for (uint j4 = 0u; j4 < dpad4; ++j4) {
                pv[j4] = fma(sc4, fv[j4], pv[j4]);
            }
        }
    }

    // 4. New Hamiltonian (reuse fbuf).
    float U_new = 0.0f;
    for (uint i = 0u; i < d; ++i) U_new = fma(0.5f * qbuf[i], fbuf[i], U_new);
    float K_new = 0.0f;
    for (uint i = 0u; i < d; ++i) K_new = fma(0.5f * pbuf[i], pbuf[i], K_new);

    // 5. Accept/reject.
    float dH = (U_new + K_new) - (U_old + K_old);
    uint b_acc = rand_u32(seed, chain_idx, base_counter + d);
    float u_acc = u01_from_bits(b_acc);
    float log_u = log(max(u_acc, 1.0e-30f));
    bool accept = isfinite(dH) && (log_u < -dH);

    for (uint i = 0u; i < d; ++i) {
        q_out[chain_idx * d + i] = accept ? qbuf[i] : qold[i];
    }
    if (accept) {
        accept_cnt[chain_idx] = accept_cnt[chain_idx] + 1u;
    }
}
```

Incumbent result:
         d8_K16384: correct, 35.72 ms, 121.1 GFLOPS (mat-vec FMAs only) (2.7% of 4500 GFLOPS)
         d16_K4096: correct, 43.25 ms, 103.3 GFLOPS (mat-vec FMAs only) (2.3% of 4500 GFLOPS)
         d32_K1024: correct, 101.98 ms, 44.5 GFLOPS (mat-vec FMAs only) (1.0% of 4500 GFLOPS)
  score (gmean of fraction): 0.0183

## History

- iter  0: compile=OK | correct=True | score=0.008820980244713755
- iter  1: compile=OK | correct=True | score=0.008896577055895305
- iter  2: compile=OK | correct=True | score=0.018287045685874234
- iter  3: compile=OK | correct=True | score=0.0034231485888168525
- iter  4: compile=OK | correct=True | score=0.017603216634095798

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
