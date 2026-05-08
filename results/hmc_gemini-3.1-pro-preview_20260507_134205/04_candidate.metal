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