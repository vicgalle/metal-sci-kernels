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