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

inline void mat_madd(thread simdgroup_float8x8& D, thread const simdgroup_float8x8& S, float scale) {
    for (uint e = 0; e < 2; ++e) {
        D.thread_elements()[e] += scale * S.thread_elements()[e];
    }
}

inline void compute_force(thread simdgroup_float8x8 F[4][4],
                          thread const simdgroup_float8x8 Q[4][4],
                          thread const simdgroup_float8x8 A[4][4],
                          uint nb) {
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < nb; ++j) {
            F[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint k = 0; k < nb; ++k) {
                simdgroup_multiply_accumulate(F[i][j], Q[i][k], A[k][j], F[i][j]);
            }
        }
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
                     uint tid       [[thread_position_in_threadgroup]],
                     uint tg_size   [[threads_per_threadgroup]]) 
                 [[max_total_threads_per_threadgroup(64)]]
{
    uint sg_id = tid / 32;
    uint lane_id = tid % 32;
    uint nb = (d + 7) / 8;
    
    threadgroup float tg_exchange[2][32][32];
    threadgroup float tg_A[32][32];

    // Initialize tg_A to zero
    for (uint idx = tid; idx < 1024; idx += tg_size) {
        tg_A[idx / 32][idx % 32] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load precision matrix A
    for (uint idx = tid; idx < d * d; idx += tg_size) {
        uint r = idx / d;
        uint c = idx % d;
        tg_A[r][c] = A[idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 A_mat[4][4];
    for (uint i = 0; i < nb; ++i) {
        for (uint j = 0; j < nb; ++j) {
            simdgroup_load(A_mat[i][j], &tg_A[i*8][j*8], 32);
        }
    }

    float q_local[32];
    float p_local[32];
    for (uint i = 0; i < 32; ++i) {
        q_local[i] = 0.0f;
        p_local[i] = 0.0f;
    }

    uint base_counter = hmc_step_idx * (d + 1);
    uint thread_seed = seed + chain_idx * 0x9E3779B9u;

    if (chain_idx < K) {
        for (uint i = 0; i < d; ++i) {
            q_local[i] = q_in[chain_idx * d + i];
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
            p_local[i]   = r * c_val;
            p_local[i+1] = r * s_val;
        }
    }

    simdgroup_float8x8 Q_mat[4][4];
    simdgroup_float8x8 P_mat[4][4];
    simdgroup_float8x8 F_mat[4][4];
    
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 4; ++j) {
            Q_mat[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
            P_mat[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    // Zero out exchange cache
    for (uint i = 0; i < 32; ++i) {
        tg_exchange[sg_id][lane_id][i] = 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Stage Q to SIMD matrices
    for (uint i = 0; i < d; ++i) {
        tg_exchange[sg_id][lane_id][i] = q_local[i];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < nb; ++j) {
            simdgroup_load(Q_mat[i][j], &tg_exchange[sg_id][i*8][j*8], 32);
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Stage P to SIMD matrices
    for (uint i = 0; i < d; ++i) {
        tg_exchange[sg_id][lane_id][i] = p_local[i];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < nb; ++j) {
            simdgroup_load(P_mat[i][j], &tg_exchange[sg_id][i*8][j*8], 32);
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Initial Force
    compute_force(F_mat, Q_mat, A_mat, nb);

    // Save out initial force
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < nb; ++j) {
            simdgroup_store(F_mat[i][j], &tg_exchange[sg_id][i*8][j*8], 32);
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    float f_local_old[32];
    for (uint i = 0; i < d; ++i) {
        f_local_old[i] = tg_exchange[sg_id][lane_id][i];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    float U_old = 0.0f;
    float K_old = 0.0f;
    for (uint i = 0; i < d; ++i) {
        U_old += 0.5f * q_local[i] * f_local_old[i];
        K_old += 0.5f * p_local[i] * p_local[i];
    }

    // Leapfrog initial kick
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < nb; ++j) {
            mat_madd(P_mat[i][j], F_mat[i][j], -0.5f * eps);
        }
    }

    if (L > 0) {
        for (uint l = 0; l < L - 1; ++l) {
            // Drift
            for (uint i = 0; i < 4; ++i) {
                for (uint j = 0; j < nb; ++j) {
                    mat_madd(Q_mat[i][j], P_mat[i][j], eps);
                }
            }
            // Recompute force
            compute_force(F_mat, Q_mat, A_mat, nb);
            // Kick
            for (uint i = 0; i < 4; ++i) {
                for (uint j = 0; j < nb; ++j) {
                    mat_madd(P_mat[i][j], F_mat[i][j], -eps);
                }
            }
        }

        // Final drift
        for (uint i = 0; i < 4; ++i) {
            for (uint j = 0; j < nb; ++j) {
                mat_madd(Q_mat[i][j], P_mat[i][j], eps);
            }
        }
        // Final force
        compute_force(F_mat, Q_mat, A_mat, nb);
        // Final half-kick
        for (uint i = 0; i < 4; ++i) {
            for (uint j = 0; j < nb; ++j) {
                mat_madd(P_mat[i][j], F_mat[i][j], -0.5f * eps);
            }
        }
    }

    float q_new[32];
    float p_new[32];
    float f_new[32];
    float U_new = U_old;
    float K_new = K_old;

    if (L > 0) {
        // Read out Q
        for (uint i = 0; i < 4; ++i) {
            for (uint j = 0; j < nb; ++j) {
                simdgroup_store(Q_mat[i][j], &tg_exchange[sg_id][i*8][j*8], 32);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i = 0; i < d; ++i) q_new[i] = tg_exchange[sg_id][lane_id][i];
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Read out P
        for (uint i = 0; i < 4; ++i) {
            for (uint j = 0; j < nb; ++j) {
                simdgroup_store(P_mat[i][j], &tg_exchange[sg_id][i*8][j*8], 32);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i = 0; i < d; ++i) p_new[i] = tg_exchange[sg_id][lane_id][i];
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Read out F
        for (uint i = 0; i < 4; ++i) {
            for (uint j = 0; j < nb; ++j) {
                simdgroup_store(F_mat[i][j], &tg_exchange[sg_id][i*8][j*8], 32);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i = 0; i < d; ++i) f_new[i] = tg_exchange[sg_id][lane_id][i];

        U_new = 0.0f;
        K_new = 0.0f;
        for (uint i = 0; i < d; ++i) {
            U_new += 0.5f * q_new[i] * f_new[i];
            K_new += 0.5f * p_new[i] * p_new[i];
        }
    } else {
        for (uint i = 0; i < d; ++i) {
            q_new[i] = q_local[i];
        }
    }

    // Accept / Reject
    float dH = (U_new + K_new) - (U_old + K_old);
    uint b_acc = mix32(mix32(thread_seed ^ (base_counter + d)));
    float u_acc = float(b_acc >> 8) * (1.0f / 16777216.0f);
    float log_u = log(max(u_acc, 1.0e-30f));
    bool accept = isfinite(dH) && (log_u < -dH);

    if (chain_idx < K) {
        for (uint i = 0; i < d; ++i) {
            q_out[chain_idx * d + i] = accept ? q_new[i] : q_local[i];
        }
        if (accept) {
            accept_cnt[chain_idx] += 1u;
        }
    }
}
```

Result of previous attempt:
  COMPILE FAILED: Error Domain=MTLLibraryErrorDomain Code=3 "program_source:43:20: error: 'max_total_threads_per_threadgroup' attribute cannot be applied to types
                 [[max_total_threads_per_threadgroup(64)]]
                   ^
" UserInfo={NSLocalizedDescription=program_source:43:20: error: 'max_total_threads_per_threadgroup' attribute cannot be applied to types
                 [[max_total_threads_per_threadgroup(64)]]
                   ^
}

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

inline uint mix32(uint x) {
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    return x ^ (x >> 16);
}

template <uint D_HALF>
inline void hmc_core(
    uint chain_idx, uint d, uint L, float eps, uint base_counter, uint thread_seed,
    device const float2* q_in_vec, device float2* q_out_vec, device uint* accept_cnt,
    threadgroup float2* A_shared)
{
    float2 q[D_HALF];
    float2 p[D_HALF];

    // 1. Momentum p ~ N(0, I)
    #pragma clang loop unroll(full)
    for (uint i = 0; i < D_HALF; ++i) {
        uint b1 = mix32(mix32(thread_seed ^ (base_counter + 2 * i)));
        uint b2 = mix32(mix32(thread_seed ^ (base_counter + 2 * i + 1)));
        float u1 = max(float(b1 >> 8) * (1.0f / 16777216.0f), 1.0e-7f);
        float u2 = float(b2 >> 8) * (1.0f / 16777216.0f);
        float r = sqrt(-2.0f * log(u1));
        float angle = 6.2831853071795864f * u2;
        float c, s;
        s = sincos(angle, c);
        p[i] = float2(r * c, r * s);
    }

    // 2. Load q
    #pragma clang loop unroll(full)
    for (uint i = 0; i < D_HALF; ++i) {
        q[i] = q_in_vec[chain_idx * D_HALF + i];
    }

    // 3. Compute U_old, K_old, and initial force
    float U_old = 0.0f;
    float K_old = 0.0f;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < D_HALF; ++i) {
        uint row0 = 2 * i;
        uint row1 = 2 * i + 1;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < D_HALF; ++j) {
            float2 qj = q[j];
            acc0 += dot(A_shared[row0 * D_HALF + j], qj);
            acc1 += dot(A_shared[row1 * D_HALF + j], qj);
        }
        float2 fi = float2(acc0, acc1);
        U_old += 0.5f * dot(q[i], fi);
        K_old += 0.5f * dot(p[i], p[i]);
        p[i] -= 0.5f * eps * fi;
    }

    float U_new = 0.0f;
    
    // 4. Leapfrog
    for (uint l = 0; l < L - 1; ++l) {
        // drift
        #pragma clang loop unroll(full)
        for (uint i = 0; i < D_HALF; ++i) {
            q[i] += eps * p[i];
        }
        
        // recompute force and kick
        #pragma clang loop unroll(full)
        for (uint i = 0; i < D_HALF; ++i) {
            uint row0 = 2 * i;
            uint row1 = 2 * i + 1;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            #pragma clang loop unroll(full)
            for (uint j = 0; j < D_HALF; ++j) {
                float2 qj = q[j];
                acc0 += dot(A_shared[row0 * D_HALF + j], qj);
                acc1 += dot(A_shared[row1 * D_HALF + j], qj);
            }
            p[i] -= eps * float2(acc0, acc1);
        }
    }

    // Last iteration (l = L - 1)
    if (L > 0) {
        // drift
        #pragma clang loop unroll(full)
        for (uint i = 0; i < D_HALF; ++i) {
            q[i] += eps * p[i];
        }
        
        // recompute force and kick
        #pragma clang loop unroll(full)
        for (uint i = 0; i < D_HALF; ++i) {
            uint row0 = 2 * i;
            uint row1 = 2 * i + 1;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            #pragma clang loop unroll(full)
            for (uint j = 0; j < D_HALF; ++j) {
                float2 qj = q[j];
                acc0 += dot(A_shared[row0 * D_HALF + j], qj);
                acc1 += dot(A_shared[row1 * D_HALF + j], qj);
            }
            float2 fi = float2(acc0, acc1);
            p[i] -= (0.5f * eps) * fi;
            U_new += 0.5f * dot(q[i], fi);
        }
    } else {
        U_new = U_old;
    }

    float K_new = 0.0f;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < D_HALF; ++i) {
        K_new += 0.5f * dot(p[i], p[i]);
    }

    // 5. Accept logic
    float dH = (U_new + K_new) - (U_old + K_old);
    uint b_acc = mix32(mix32(thread_seed ^ (base_counter + d)));
    float u_acc = float(b_acc >> 8) * (1.0f / 16777216.0f);
    float log_u = log(max(u_acc, 1.0e-30f));
    bool accept = isfinite(dH) && (log_u < -dH);

    #pragma clang loop unroll(full)
    for (uint i = 0; i < D_HALF; ++i) {
        q_out_vec[chain_idx * D_HALF + i] = accept ? q[i] : q_in_vec[chain_idx * D_HALF + i];
    }
    if (accept) {
        accept_cnt[chain_idx] += 1u;
    }
}

#define HMC_CORE_CASE(n) \
    case n: hmc_core<n>(chain_idx, d, L, eps, base_counter, thread_seed, q_in_vec, q_out_vec, accept_cnt, A_shared); break;

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
    // Max d=32 -> 1024 floats -> 512 float2s
    threadgroup float2 A_shared[512];
    
    // Cooperatively load matrix A into fast threadgroup cache.
    uint d_sq_half = (d * d) / 2;
    device const float2* A_vec = (device const float2*)A;
    for(uint i = tid_in_tg; i < d_sq_half; i += tg_size) {
        A_shared[i] = A_vec[i];
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (chain_idx >= K) return;
    
    uint d_half = d / 2;
    uint base_counter = hmc_step_idx * (d + 1u);
    uint thread_seed = seed + chain_idx * 0x9E3779B9u;
    
    device const float2* q_in_vec = (device const float2*)q_in;
    device float2* q_out_vec = (device float2*)q_out;

    switch (d_half) {
        HMC_CORE_CASE(1)
        HMC_CORE_CASE(2)
        HMC_CORE_CASE(3)
        HMC_CORE_CASE(4)
        HMC_CORE_CASE(5)
        HMC_CORE_CASE(6)
        HMC_CORE_CASE(7)
        HMC_CORE_CASE(8)
        HMC_CORE_CASE(9)
        HMC_CORE_CASE(10)
        HMC_CORE_CASE(11)
        HMC_CORE_CASE(12)
        HMC_CORE_CASE(13)
        HMC_CORE_CASE(14)
        HMC_CORE_CASE(15)
        HMC_CORE_CASE(16)
    }
}
```

Incumbent result:
         d8_K16384: correct, 4.69 ms, 922.6 GFLOPS (mat-vec FMAs only) (20.5% of 4500 GFLOPS)
         d16_K4096: correct, 40.89 ms, 109.3 GFLOPS (mat-vec FMAs only) (2.4% of 4500 GFLOPS)
         d32_K1024: correct, 179.97 ms, 25.2 GFLOPS (mat-vec FMAs only) (0.6% of 4500 GFLOPS)
  score (gmean of fraction): 0.0303

## History

- iter  0: compile=OK | correct=True | score=0.008135524682384003
- iter  1: compile=OK | correct=True | score=0.023622290244432038
- iter  2: compile=OK | correct=True | score=0.030339903447087622
- iter  3: compile=FAIL | correct=False | score=N/A
- iter  4: compile=FAIL | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
