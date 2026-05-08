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

template <uint D8>
inline float compute_force_and_energy(
    thread float* p,
    float p_scale,
    thread const float* q,
    uint d,
    uint sg_id,
    uint lane_id,
    threadgroup float A_tg[32][32],
    threadgroup float Q_tg[2][32][32])
{
    // Coalesced write of q to threadgroup memory
    #pragma clang loop unroll(full)
    for (uint i = 0; i < D8 * 8; ++i) {
        Q_tg[sg_id][i][lane_id] = q[i];
    }
    
    simdgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 F_mat[4][4];
    #pragma clang loop unroll(full)
    for (uint r = 0; r < D8; ++r) {
        #pragma clang loop unroll(full)
        for (uint c = 0; c < 4; ++c) {
            F_mat[r][c] = simdgroup_float8x8(0.0f);
        }
    }

    // Multiply-accumulate using AMX blocks
    #pragma clang loop unroll(full)
    for (uint k = 0; k < D8; ++k) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < D8; ++r) {
            simdgroup_float8x8 A_mat;
            simdgroup_load(A_mat, &A_tg[r*8][k*8], 32);
            
            #pragma clang loop unroll(full)
            for (uint c = 0; c < 4; ++c) {
                simdgroup_float8x8 Q_mat;
                simdgroup_load(Q_mat, &Q_tg[sg_id][k*8][c*8], 32);
                simdgroup_multiply_accumulate(F_mat[r][c], A_mat, Q_mat, F_mat[r][c]);
            }
        }
    }
    
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Store resultant forces back to threadgroup
    #pragma clang loop unroll(full)
    for (uint r = 0; r < D8; ++r) {
        #pragma clang loop unroll(full)
        for (uint c = 0; c < 4; ++c) {
            simdgroup_store(F_mat[r][c], &Q_tg[sg_id][r*8][c*8], 32);
        }
    }

    simdgroup_barrier(mem_flags::mem_threadgroup);

    float U = 0.0f;
    // Compute partial energy and update p
    #pragma clang loop unroll(full)
    for (uint i = 0; i < D8 * 8; ++i) {
        float f_val = Q_tg[sg_id][i][lane_id];
        if (i < d) {
            U += 0.5f * q[i] * f_val;
        }
        p[i] -= p_scale * f_val;
    }
    
    simdgroup_barrier(mem_flags::mem_threadgroup);
    
    return U;
}

template <uint D8>
inline void hmc_core(
    uint chain_idx, uint d, uint K, uint L, float eps, uint base_counter, uint thread_seed,
    device const float* q_in, device float* q_out, device uint* accept_cnt,
    threadgroup float A_tg[32][32], threadgroup float Q_tg[2][32][32],
    uint sg_id, uint lane_id)
{
    float q[D8 * 8];
    float p[D8 * 8];

    #pragma clang loop unroll(full)
    for (uint i = 0; i < D8 * 8; ++i) {
        q[i] = 0.0f;
        p[i] = 0.0f;
    }

    // 1) p ~ N(0, I)
    for (uint i = 0; i < d; i += 2) {
        uint b1 = mix32(mix32(thread_seed ^ (base_counter + i)));
        uint b2 = mix32(mix32(thread_seed ^ (base_counter + i + 1)));
        float u1 = max(float(b1 >> 8) * (1.0f / 16777216.0f), 1.0e-7f);
        float u2 = float(b2 >> 8) * (1.0f / 16777216.0f);
        float r = sqrt(-2.0f * log(u1));
        float angle = 6.2831853071795864f * u2;
        p[i] = r * cos(angle);
        if (i + 1 < d) {
            p[i+1] = r * sin(angle);
        }
    }

    bool valid = (chain_idx < K);
    for (uint i = 0; i < D8 * 8; ++i) {
        if (valid && i < d) {
            q[i] = q_in[chain_idx * d + i];
        }
    }

    // 2) Save K_old
    float K_old = 0.0f;
    for (uint i = 0; i < d; ++i) {
        K_old += 0.5f * p[i] * p[i];
    }

    // 3.1) Compute initial force, U_old, and initial half-kick
    float U_old = compute_force_and_energy<D8>(p, 0.5f * eps, q, d, sg_id, lane_id, A_tg, Q_tg);
    float U_new = U_old;

    // 3.2) Leapfrog
    for (uint l = 0; l < L; ++l) {
        #pragma clang loop unroll(full)
        for (uint i = 0; i < D8 * 8; ++i) {
            q[i] += eps * p[i];
        }

        float scale = (l + 1 == L) ? (0.5f * eps) : eps;
        U_new = compute_force_and_energy<D8>(p, scale, q, d, sg_id, lane_id, A_tg, Q_tg);
    }

    // 4) K_new
    float K_new = 0.0f;
    for (uint i = 0; i < d; ++i) {
        K_new += 0.5f * p[i] * p[i];
    }

    float dH = (U_new + K_new) - (U_old + K_old);

    // 5) Accept/Reject
    uint b_acc = mix32(mix32(thread_seed ^ (base_counter + d)));
    float u_acc = float(b_acc >> 8) * (1.0f / 16777216.0f);
    float log_u = log(max(u_acc, 1.0e-30f));
    bool accept = isfinite(dH) && (log_u < -dH);

    if (valid) {
        for (uint i = 0; i < d; ++i) {
            q_out[chain_idx * d + i] = accept ? q[i] : q_in[chain_idx * d + i];
        }
        if (accept) {
            accept_cnt[chain_idx] += 1u;
        }
    }
}

[[max_total_threads_per_threadgroup(64)]]
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
    threadgroup float A_tg[32][32];
    threadgroup float Q_tg[2][32][32]; // Accommodates up to 64 threads per TG

    uint D_pad = (d + 7) & ~7u;

    // Load constant A matrix uniformly into threadgroup memory
    for (uint idx = tid_in_tg; idx < D_pad * D_pad; idx += tg_size) {
        uint r = idx / D_pad;
        uint c = idx % D_pad;
        A_tg[r][c] = (r < d && c < d) ? A[r * d + c] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sg_id = tid_in_tg / 32;
    uint lane_id = tid_in_tg % 32;

    uint base_counter = hmc_step_idx * (d + 1u);
    uint thread_seed = seed + chain_idx * 0x9E3779B9u;

    uint D8 = D_pad / 8;
    switch (D8) {
        case 1: hmc_core<1>(chain_idx, d, K, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A_tg, Q_tg, sg_id, lane_id); break;
        case 2: hmc_core<2>(chain_idx, d, K, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A_tg, Q_tg, sg_id, lane_id); break;
        case 3: hmc_core<3>(chain_idx, d, K, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A_tg, Q_tg, sg_id, lane_id); break;
        case 4: hmc_core<4>(chain_idx, d, K, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A_tg, Q_tg, sg_id, lane_id); break;
    }
}
```

Result of previous attempt:
         d8_K16384: correct, 11.12 ms, 389.1 GFLOPS (mat-vec FMAs only) (8.6% of 4500 GFLOPS)
         d16_K4096: correct, 6.63 ms, 674.2 GFLOPS (mat-vec FMAs only) (15.0% of 4500 GFLOPS)
         d32_K1024: correct, 19.82 ms, 229.1 GFLOPS (mat-vec FMAs only) (5.1% of 4500 GFLOPS)
  score (gmean of fraction): 0.0870

## History

- iter  2: compile=OK | correct=True | score=0.030339903447087622
- iter  3: compile=FAIL | correct=False | score=N/A
- iter  4: compile=FAIL | correct=False | score=N/A
- iter  5: compile=OK | correct=True | score=0.00686007131718383
- iter  6: compile=OK | correct=True | score=0.034084391906103566
- iter  7: compile=OK | correct=True | score=0.03202151274908618
- iter  8: compile=OK | correct=True | score=0.04646230219205438
- iter  9: compile=OK | correct=True | score=0.0870428494203685

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
