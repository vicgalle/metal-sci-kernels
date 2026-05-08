To dramatically improve performance, we replace the fully unrolled per-thread matrix-vector multiplications with cooperative SIMD-group matrix-matrix multiplications (`simdgroup_matrix`). By treating the positions of 32 chains (one full SIMD group) as a single $32 \times d$ matrix $Q$, the force computation $F = Q A$ can be executed natively on Apple Silicon's AMX tensor cores. 

This converts the previously memory-bound $O(d^2)$ inner leapfrog loop into a blisteringly fast AMX matmul, heavily reducing instruction count and preventing register spilling. To stay safely within the 32 KB threadgroup memory limit regardless of host threadgroup size, we serialize the matrix staging in threadgroup memory across SIMD groups, strictly bounding its footprint to just 8 KB.

```metal
#include <metal_stdlib>
using namespace metal;

inline uint mix32(uint x) {
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    return x ^ (x >> 16);
}

inline void stage_load_mat(thread simdgroup_float8x8 M[4][4],
                           thread const float src[32],
                           threadgroup float tg_Q[32][32],
                           uint sg_id, uint lane_id, uint num_sgs, uint d, uint num_blocks_d) {
    for (uint sg = 0; sg < num_sgs; ++sg) {
        if (sg_id == sg) {
            for (uint i = 0; i < 32; ++i) {
                tg_Q[lane_id][i] = (i < d) ? src[i] : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == sg) {
            for (uint i = 0; i < 4; ++i) {
                for (uint j = 0; j < num_blocks_d; ++j) {
                    simdgroup_load(M[i][j], &tg_Q[i * 8][j * 8], 32);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

inline void stage_store_mat(thread const simdgroup_float8x8 M[4][4],
                            thread float dst[32],
                            threadgroup float tg_Q[32][32],
                            uint sg_id, uint lane_id, uint num_sgs, uint d, uint num_blocks_d) {
    for (uint sg = 0; sg < num_sgs; ++sg) {
        if (sg_id == sg) {
            for (uint i = 0; i < 4; ++i) {
                for (uint j = 0; j < num_blocks_d; ++j) {
                    simdgroup_store(M[i][j], &tg_Q[i * 8][j * 8], 32);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == sg) {
            for (uint i = 0; i < d; ++i) {
                dst[i] = tg_Q[lane_id][i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

inline void compute_force(thread simdgroup_float8x8 F_mat[4][4],
                          thread const simdgroup_float8x8 X_mat[4][4],
                          thread const simdgroup_float8x8 A_mat[4][4],
                          uint num_blocks_d) {
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < num_blocks_d; ++j) {
            for (uint e = 0; e < 2; ++e) {
                F_mat[i][j].thread_elements()[e] = 0.0f;
            }
            for (uint k = 0; k < num_blocks_d; ++k) {
                simdgroup_multiply_accumulate(F_mat[i][j], X_mat[i][k], A_mat[k][j]);
            }
        }
    }
}

inline void mat_madd(thread simdgroup_float8x8 D_mat[4][4],
                     thread const simdgroup_float8x8 S_mat[4][4],
                     float scale,
                     uint num_blocks_d) {
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < num_blocks_d; ++j) {
            for (uint e = 0; e < 2; ++e) {
                D_mat[i][j].thread_elements()[e] += scale * S_mat[i][j].thread_elements()[e];
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
{
    uint sg_id = tid / 32;
    uint lane_id = tid % 32;
    uint num_sgs = (tg_size + 31) / 32;
    uint num_blocks_d = (d + 7) / 8;
    
    threadgroup float tg_Q[32][32];
    threadgroup float tg_A[32][32];

    // Load precision matrix A into shared memory (padded to 32x32 to safely support all dimensions)
    for (uint idx = tid; idx < 1024; idx += tg_size) {
        tg_A[idx / 32][idx % 32] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < d * d; idx += tg_size) {
        tg_A[idx / d][idx % d] = A[idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 A_mat[4][4];
    for (uint i = 0; i < num_blocks_d; ++i) {
        for (uint j = 0; j < num_blocks_d; ++j) {
            simdgroup_load(A_mat[i][j], &tg_A[i * 8][j * 8], 32);
        }
    }

    // Prepare momentum, initial points, and force buffers per thread
    float q_old[32];
    float p_local[32];
    for (uint i = 0; i < 32; ++i) {
        q_old[i] = 0.0f;
        p_local[i] = 0.0f;
    }

    uint base_counter = hmc_step_idx * (d + 1);
    uint thread_seed = (chain_idx < K) ? (seed + chain_idx * 0x9E3779B9u) : 0u;

    if (chain_idx < K) {
        for (uint i = 0; i < d; ++i) {
            q_old[i] = q_in[chain_idx * d + i];
        }

        // Draw initial momentum
        for (uint i = 0; i < d; i += 2) {
            uint b1 = mix32(mix32(thread_seed ^ (base_counter + i)));
            uint b2 = mix32(mix32(thread_seed ^ (base_counter + i + 1)));
            float u1 = max(float(b1 >> 8) * (1.0f / 16777216.0f), 1.0e-7f);
            float u2 = float(b2 >> 8) * (1.0f / 16777216.0f);
            float r = sqrt(-2.0f * log(u1));
            float angle = 6.2831853071795864f * u2;
            float c, s;
            s = sincos(angle, c);
            p_local[i] = r * c;
            p_local[i + 1] = r * s;
        }
    }

    // Load initialized elements into SIMD group matrices
    simdgroup_float8x8 X_mat[4][4];
    simdgroup_float8x8 P_mat[4][4];
    simdgroup_float8x8 F_mat[4][4];
    
    stage_load_mat(X_mat, q_old, tg_Q, sg_id, lane_id, num_sgs, d, num_blocks_d);
    stage_load_mat(P_mat, p_local, tg_Q, sg_id, lane_id, num_sgs, d, num_blocks_d);

    // Initial energy and kick logic
    compute_force(F_mat, X_mat, A_mat, num_blocks_d);
    
    float force_local[32];
    stage_store_mat(F_mat, force_local, tg_Q, sg_id, lane_id, num_sgs, d, num_blocks_d);

    float U_old = 0.0f;
    float K_old = 0.0f;
    for (uint i = 0; i < d; ++i) {
        U_old += 0.5f * q_old[i] * force_local[i];
        K_old += 0.5f * p_local[i] * p_local[i];
    }

    mat_madd(P_mat, F_mat, -0.5f * eps, num_blocks_d);

    float U_new = 0.0f;
    float K_new = 0.0f;
    float q_new[32];

    if (L > 0) {
        for (uint l = 0; l < L - 1; ++l) {
            mat_madd(X_mat, P_mat, eps, num_blocks_d);
            compute_force(F_mat, X_mat, A_mat, num_blocks_d);
            mat_madd(P_mat, F_mat, -eps, num_blocks_d);
        }

        // Final leapfrog step
        mat_madd(X_mat, P_mat, eps, num_blocks_d);
        compute_force(F_mat, X_mat, A_mat, num_blocks_d);
        mat_madd(P_mat, F_mat, -0.5f * eps, num_blocks_d);

        stage_store_mat(X_mat, q_new, tg_Q, sg_id, lane_id, num_sgs, d, num_blocks_d);
        
        float p_new[32];
        stage_store_mat(P_mat, p_new, tg_Q, sg_id, lane_id, num_sgs, d, num_blocks_d);
        
        float force_new[32];
        stage_store_mat(F_mat, force_new, tg_Q, sg_id, lane_id, num_sgs, d, num_blocks_d);

        for (uint i = 0; i < d; ++i) {
            U_new += 0.5f * q_new[i] * force_new[i];
            K_new += 0.5f * p_new[i] * p_new[i];
        }
    } else {
        U_new = U_old;
        K_new = K_old;
        for (uint i = 0; i < d; ++i) {
            q_new[i] = q_old[i];
        }
    }

    // Acceptance logic
    float dH = (U_new + K_new) - (U_old + K_old);
    uint b_acc = mix32(mix32(thread_seed ^ (base_counter + d)));
    float u_acc = float(b_acc >> 8) * (1.0f / 16777216.0f);
    float log_u = log(max(u_acc, 1.0e-30f));
    bool accept = isfinite(dH) && (log_u < -dH);

    if (chain_idx < K) {
        for (uint i = 0; i < d; ++i) {
            q_out[chain_idx * d + i] = accept ? q_new[i] : q_old[i];
        }
        if (accept) {
            accept_cnt[chain_idx] += 1u;
        }
    }
}
```