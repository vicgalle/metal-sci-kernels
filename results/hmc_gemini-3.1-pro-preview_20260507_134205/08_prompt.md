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
    return x ^ (x >> 16);
}

template <uint D>
inline void hmc_core(
    uint chain_idx, uint L, float eps, uint base_counter, uint thread_seed,
    device const float* q_in, device float* q_out, device uint* accept_cnt,
    device const float* A)
{
    float2 q2[D/2];
    float2 p2[D/2];

    device const float2* q_in_2 = (device const float2*)q_in;
    device const float2* A_2 = (device const float2*)A;

    #pragma clang loop unroll(full)
    for (uint i = 0; i < D/2; ++i) {
        q2[i] = q_in_2[chain_idx * (D/2) + i];
    }

    #pragma clang loop unroll(full)
    for (uint i = 0; i < D/2; ++i) {
        uint b1 = mix32(mix32(thread_seed ^ (base_counter + i * 2)));
        uint b2 = mix32(mix32(thread_seed ^ (base_counter + i * 2 + 1)));
        float u1 = max(float(b1 >> 8) * (1.0f / 16777216.0f), 1.0e-7f);
        float u2 = float(b2 >> 8) * (1.0f / 16777216.0f);
        float r = sqrt(-2.0f * log(u1));
        float angle = 6.2831853071795864f * u2;
        float c_val, s_val;
        s_val = sincos(angle, c_val);
        
        p2[i] = float2(r * c_val, r * s_val);
    }

    float U_old = 0.0f;
    float K_old = 0.0f;
    
    #pragma clang loop unroll(full)
    for (uint i = 0; i < D/2; ++i) {
        K_old += 0.5f * dot(p2[i], p2[i]);
    }
    
    #pragma clang loop unroll(full)
    for (uint i = 0; i < D/2; ++i) {
        float f_acc0 = 0.0f;
        float f_acc1 = 0.0f;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < D/2; ++j) {
            float2 q_val = q2[j];
            f_acc0 += dot(A_2[(i * 2 + 0) * (D/2) + j], q_val);
            f_acc1 += dot(A_2[(i * 2 + 1) * (D/2) + j], q_val);
        }
        U_old += 0.5f * dot(q2[i], float2(f_acc0, f_acc1));
        p2[i] -= 0.5f * eps * float2(f_acc0, f_acc1);
    }

    for (uint l = 0; l + 1 < L; ++l) {
        #pragma clang loop unroll(full)
        for (uint i = 0; i < D/2; ++i) {
            q2[i] += eps * p2[i];
        }
        
        #pragma clang loop unroll(full)
        for (uint i = 0; i < D/2; ++i) {
            float f_acc0 = 0.0f;
            float f_acc1 = 0.0f;
            #pragma clang loop unroll(full)
            for (uint j = 0; j < D/2; ++j) {
                float2 q_val = q2[j];
                f_acc0 += dot(A_2[(i * 2 + 0) * (D/2) + j], q_val);
                f_acc1 += dot(A_2[(i * 2 + 1) * (D/2) + j], q_val);
            }
            p2[i] -= eps * float2(f_acc0, f_acc1);
        }
    }

    float U_new = 0.0f;
    if (L > 0) {
        #pragma clang loop unroll(full)
        for (uint i = 0; i < D/2; ++i) {
            q2[i] += eps * p2[i];
        }
        
        #pragma clang loop unroll(full)
        for (uint i = 0; i < D/2; ++i) {
            float f_acc0 = 0.0f;
            float f_acc1 = 0.0f;
            #pragma clang loop unroll(full)
            for (uint j = 0; j < D/2; ++j) {
                float2 q_val = q2[j];
                f_acc0 += dot(A_2[(i * 2 + 0) * (D/2) + j], q_val);
                f_acc1 += dot(A_2[(i * 2 + 1) * (D/2) + j], q_val);
            }
            U_new += 0.5f * dot(q2[i], float2(f_acc0, f_acc1));
            p2[i] -= 0.5f * eps * float2(f_acc0, f_acc1);
        }
    } else {
        U_new = U_old;
    }

    float K_new = 0.0f;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < D/2; ++i) {
        K_new += 0.5f * dot(p2[i], p2[i]);
    }

    float dH = (U_new + K_new) - (U_old + K_old);
    uint b_acc = mix32(mix32(thread_seed ^ (base_counter + D)));
    float u_acc = float(b_acc >> 8) * (1.0f / 16777216.0f);
    float log_u = log(max(u_acc, 1.0e-30f));
    bool accept = isfinite(dH) && (log_u < -dH);

    device float2* q_out_2 = (device float2*)q_out;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < D/2; ++i) {
        q_out_2[chain_idx * (D/2) + i] = accept ? q2[i] : q_in_2[chain_idx * (D/2) + i];
    }
    
    if (accept) {
        accept_cnt[chain_idx] += 1u;
    }
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
                     uint chain_idx [[thread_position_in_grid]])
{
    if (chain_idx >= K) return;
    
    uint base_counter = hmc_step_idx * (d + 1u);
    uint thread_seed = seed + chain_idx * 0x9E3779B9u;
    
    switch (d) {
        case 2: hmc_core<2>(chain_idx, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A); break;
        case 4: hmc_core<4>(chain_idx, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A); break;
        case 6: hmc_core<6>(chain_idx, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A); break;
        case 8: hmc_core<8>(chain_idx, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A); break;
        case 10: hmc_core<10>(chain_idx, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A); break;
        case 12: hmc_core<12>(chain_idx, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A); break;
        case 14: hmc_core<14>(chain_idx, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A); break;
        case 16: hmc_core<16>(chain_idx, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A); break;
        case 18: hmc_core<18>(chain_idx, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A); break;
        case 20: hmc_core<20>(chain_idx, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A); break;
        case 22: hmc_core<22>(chain_idx, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A); break;
        case 24: hmc_core<24>(chain_idx, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A); break;
        case 26: hmc_core<26>(chain_idx, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A); break;
        case 28: hmc_core<28>(chain_idx, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A); break;
        case 30: hmc_core<30>(chain_idx, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A); break;
        case 32: hmc_core<32>(chain_idx, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A); break;
    }
}
```

Result of previous attempt:
         d8_K16384: correct, 5.95 ms, 727.3 GFLOPS (mat-vec FMAs only) (16.2% of 4500 GFLOPS)
         d16_K4096: correct, 38.74 ms, 115.4 GFLOPS (mat-vec FMAs only) (2.6% of 4500 GFLOPS)
         d32_K1024: correct, 127.37 ms, 35.7 GFLOPS (mat-vec FMAs only) (0.8% of 4500 GFLOPS)
  score (gmean of fraction): 0.0320

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

inline uint mix32(uint x) {
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    return x ^ (x >> 16);
}

template <uint D4>
inline void compute_force(thread float4* force, thread const float4* q, threadgroup const float4* A_shared) {
    #pragma clang loop unroll(full)
    for (uint i = 0; i < D4; ++i) {
        float4 f_acc = float4(0.0f);
        #pragma clang loop unroll(full)
        for (uint j = 0; j < D4; ++j) {
            float4 q_val = q[j];
            f_acc[0] += dot(A_shared[(i * 4 + 0) * D4 + j], q_val);
            f_acc[1] += dot(A_shared[(i * 4 + 1) * D4 + j], q_val);
            f_acc[2] += dot(A_shared[(i * 4 + 2) * D4 + j], q_val);
            f_acc[3] += dot(A_shared[(i * 4 + 3) * D4 + j], q_val);
        }
        force[i] = f_acc;
    }
}

template <uint D4>
inline void hmc_core(
    uint chain_idx, uint d, uint L, float eps, uint base_counter, uint thread_seed,
    device const float* q_in, device float* q_out, device uint* accept_cnt,
    threadgroup const float4* A_shared)
{
    float4 q[D4];
    float4 p[D4];
    float4 force[D4];

    #pragma clang loop unroll(full)
    for (uint i = 0; i < D4; ++i) {
        q[i] = float4(0.0f);
        p[i] = float4(0.0f);
    }

    for (uint i = 0; i < d; i += 2) {
        uint b1 = mix32(mix32(thread_seed ^ (base_counter + i)));
        uint b2 = mix32(mix32(thread_seed ^ (base_counter + i + 1)));
        float u1 = max(float(b1 >> 8) * (1.0f / 16777216.0f), 1.0e-7f);
        float u2 = float(b2 >> 8) * (1.0f / 16777216.0f);
        float r = sqrt(-2.0f * log(u1));
        float angle = 6.2831853071795864f * u2;
        float c_val, s_val;
        s_val = sincos(angle, c_val);
        
        uint d4_idx = i / 4;
        uint sub_idx = i % 4;
        p[d4_idx][sub_idx]     = r * c_val;
        p[d4_idx][sub_idx + 1] = r * s_val;
    }

    for (uint i = 0; i < d; ++i) {
        uint d4_idx = i / 4;
        uint sub_idx = i % 4;
        q[d4_idx][sub_idx] = q_in[chain_idx * d + i];
    }

    compute_force<D4>(force, q, A_shared);

    float U_old = 0.0f;
    float K_old = 0.0f;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < D4; ++i) {
        U_old += 0.5f * dot(q[i], force[i]);
        K_old += 0.5f * dot(p[i], p[i]);
        p[i] -= 0.5f * eps * force[i];
    }

    for (uint l = 0; l + 1 < L; ++l) {
        #pragma clang loop unroll(full)
        for (uint i = 0; i < D4; ++i) {
            q[i] += eps * p[i];
        }
        compute_force<D4>(force, q, A_shared);
        #pragma clang loop unroll(full)
        for (uint i = 0; i < D4; ++i) {
            p[i] -= eps * force[i];
        }
    }

    float U_new = 0.0f;
    if (L > 0) {
        #pragma clang loop unroll(full)
        for (uint i = 0; i < D4; ++i) {
            q[i] += eps * p[i];
        }
        compute_force<D4>(force, q, A_shared);
        #pragma clang loop unroll(full)
        for (uint i = 0; i < D4; ++i) {
            p[i] -= 0.5f * eps * force[i];
            U_new += 0.5f * dot(q[i], force[i]);
        }
    } else {
        U_new = U_old;
    }

    float K_new = 0.0f;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < D4; ++i) {
        K_new += 0.5f * dot(p[i], p[i]);
    }

    float dH = (U_new + K_new) - (U_old + K_old);
    uint b_acc = mix32(mix32(thread_seed ^ (base_counter + d)));
    float u_acc = float(b_acc >> 8) * (1.0f / 16777216.0f);
    float log_u = log(max(u_acc, 1.0e-30f));
    bool accept = isfinite(dH) && (log_u < -dH);

    for (uint i = 0; i < d; ++i) {
        uint d4_idx = i / 4;
        uint sub_idx = i % 4;
        q_out[chain_idx * d + i] = accept ? q[d4_idx][sub_idx] : q_in[chain_idx * d + i];
    }
    
    if (accept) {
        accept_cnt[chain_idx] += 1u;
    }
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
                     uint tg_size   [[threads_per_threadgroup]]) 
{
    threadgroup float4 A_shared[256]; // Safely bounded exactly as maximum sizes permit
    
    uint d_pad = (d + 3) & ~3u;
    uint total_padded = d_pad * d_pad;
    
    threadgroup float* A_ptr = (threadgroup float*)A_shared;
    for (uint idx = tid_in_tg; idx < total_padded; idx += tg_size) {
        uint r = idx / d_pad;
        uint c = idx % d_pad;
        A_ptr[idx] = (r < d && c < d) ? A[r * d + c] : 0.0f;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (chain_idx >= K) return;
    
    uint base_counter = hmc_step_idx * (d + 1u);
    uint thread_seed = seed + chain_idx * 0x9E3779B9u;
    uint d4 = d_pad / 4;
    
    switch (d4) {
        case 1: hmc_core<1>(chain_idx, d, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A_shared); break;
        case 2: hmc_core<2>(chain_idx, d, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A_shared); break;
        case 3: hmc_core<3>(chain_idx, d, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A_shared); break;
        case 4: hmc_core<4>(chain_idx, d, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A_shared); break;
        case 5: hmc_core<5>(chain_idx, d, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A_shared); break;
        case 6: hmc_core<6>(chain_idx, d, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A_shared); break;
        case 7: hmc_core<7>(chain_idx, d, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A_shared); break;
        case 8: hmc_core<8>(chain_idx, d, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A_shared); break;
    }
}
```

Incumbent result:
         d8_K16384: correct, 6.79 ms, 637.3 GFLOPS (mat-vec FMAs only) (14.2% of 4500 GFLOPS)
         d16_K4096: correct, 39.18 ms, 114.1 GFLOPS (mat-vec FMAs only) (2.5% of 4500 GFLOPS)
         d32_K1024: correct, 91.50 ms, 49.6 GFLOPS (mat-vec FMAs only) (1.1% of 4500 GFLOPS)
  score (gmean of fraction): 0.0341

## History

- iter  0: compile=OK | correct=True | score=0.008135524682384003
- iter  1: compile=OK | correct=True | score=0.023622290244432038
- iter  2: compile=OK | correct=True | score=0.030339903447087622
- iter  3: compile=FAIL | correct=False | score=N/A
- iter  4: compile=FAIL | correct=False | score=N/A
- iter  5: compile=OK | correct=True | score=0.00686007131718383
- iter  6: compile=OK | correct=True | score=0.034084391906103566
- iter  7: compile=OK | correct=True | score=0.03202151274908618

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
