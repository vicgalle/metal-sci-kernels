#include <metal_stdlib>
using namespace metal;

inline uint mix32(uint x) {
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    return x ^ (x >> 16);
}

template <uint D2>
inline float compute_force_and_energy_alu(
    thread float2* p,
    float p_scale,
    thread const float2* q,
    threadgroup float2 A_tg[32][16])
{
    float U = 0.0f;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < D2; ++i) {
        float f0 = 0.0f;
        float f1 = 0.0f;
        
        #pragma clang loop unroll(full)
        for (uint j = 0; j < D2; ++j) {
            f0 += dot(A_tg[i * 2][j], q[j]);
            f1 += dot(A_tg[i * 2 + 1][j], q[j]);
        }
        
        float2 f_vec = float2(f0, f1);
        float2 q_vec = q[i];
        
        U += 0.5f * dot(q_vec, f_vec);
        p[i] -= p_scale * f_vec;
    }
    return U;
}

template <uint D2>
inline void hmc_core(
    uint chain_idx, uint K, uint L, float eps, uint base_counter, uint thread_seed,
    device const float* q_in, device float* q_out, device uint* accept_cnt,
    threadgroup float2 A_tg[32][16])
{
    float2 q[D2];
    float2 q_old[D2];
    float2 p[D2];

    // 1) p ~ N(0, I)
    #pragma clang loop unroll(full)
    for (uint i = 0; i < D2; ++i) {
        uint j = i * 2;
        uint b1 = mix32(mix32(thread_seed ^ (base_counter + j)));
        uint b2 = mix32(mix32(thread_seed ^ (base_counter + j + 1)));
        float u1 = max(float(b1 >> 8) * (1.0f / 16777216.0f), 1.0e-7f);
        float u2 = float(b2 >> 8) * (1.0f / 16777216.0f);
        float r = sqrt(-2.0f * log(u1));
        float angle = 6.2831853071795864f * u2;
        p[i] = float2(r * cos(angle), r * sin(angle));
    }

    bool valid = (chain_idx < K);
    
    // Load q and save q_old
    if (valid) {
        device const float2* q_in_ptr = (device const float2*)(q_in + chain_idx * (D2 * 2));
        #pragma clang loop unroll(full)
        for (uint i = 0; i < D2; ++i) {
            float2 val = q_in_ptr[i];
            q[i] = val;
            q_old[i] = val;
        }
    } else {
        #pragma clang loop unroll(full)
        for (uint i = 0; i < D2; ++i) {
            q[i] = float2(0.0f);
            q_old[i] = float2(0.0f);
        }
    }

    // 2) Save K_old
    float K_old = 0.0f;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < D2; ++i) {
        K_old += 0.5f * dot(p[i], p[i]);
    }

    // 3.1) Compute initial force, U_old, and initial half-kick
    float U_old = compute_force_and_energy_alu<D2>(p, 0.5f * eps, q, A_tg);
    float U_new = U_old;

    // 3.2) Leapfrog
    for (uint l = 0; l < L; ++l) {
        #pragma clang loop unroll(full)
        for (uint i = 0; i < D2; ++i) {
            q[i] += eps * p[i];
        }

        float scale = (l + 1 == L) ? (0.5f * eps) : eps;
        U_new = compute_force_and_energy_alu<D2>(p, scale, q, A_tg);
    }

    // 4) K_new
    float K_new = 0.0f;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < D2; ++i) {
        K_new += 0.5f * dot(p[i], p[i]);
    }

    float dH = (U_new + K_new) - (U_old + K_old);

    // 5) Accept/Reject
    uint b_acc = mix32(mix32(thread_seed ^ (base_counter + D2 * 2)));
    float u_acc = float(b_acc >> 8) * (1.0f / 16777216.0f);
    float log_u = log(max(u_acc, 1.0e-30f));
    bool accept = isfinite(dH) && (log_u < -dH);

    if (valid) {
        device float2* q_out_ptr = (device float2*)(q_out + chain_idx * (D2 * 2));
        if (accept) {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < D2; ++i) {
                q_out_ptr[i] = q[i];
            }
            accept_cnt[chain_idx] += 1u;
        } else {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < D2; ++i) {
                q_out_ptr[i] = q_old[i];
            }
        }
    }
}

[[max_total_threads_per_threadgroup(256)]]
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
    threadgroup float2 A_tg[32][16];

    uint D2 = d / 2;
    uint num_float2 = d * D2;
    device const float2* A_ptr = (device const float2*)A;
    
    for (uint idx = tid_in_tg; idx < num_float2; idx += tg_size) {
        uint r = idx / D2;
        uint c = idx % D2;
        A_tg[r][c] = A_ptr[idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint base_counter = hmc_step_idx * (d + 1u);
    uint thread_seed = seed + chain_idx * 0x9E3779B9u;

    switch (D2) {
        case 1:  hmc_core<1>(chain_idx, K, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A_tg); break;
        case 2:  hmc_core<2>(chain_idx, K, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A_tg); break;
        case 3:  hmc_core<3>(chain_idx, K, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A_tg); break;
        case 4:  hmc_core<4>(chain_idx, K, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A_tg); break;
        case 5:  hmc_core<5>(chain_idx, K, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A_tg); break;
        case 6:  hmc_core<6>(chain_idx, K, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A_tg); break;
        case 7:  hmc_core<7>(chain_idx, K, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A_tg); break;
        case 8:  hmc_core<8>(chain_idx, K, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A_tg); break;
        case 9:  hmc_core<9>(chain_idx, K, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A_tg); break;
        case 10: hmc_core<10>(chain_idx, K, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A_tg); break;
        case 11: hmc_core<11>(chain_idx, K, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A_tg); break;
        case 12: hmc_core<12>(chain_idx, K, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A_tg); break;
        case 13: hmc_core<13>(chain_idx, K, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A_tg); break;
        case 14: hmc_core<14>(chain_idx, K, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A_tg); break;
        case 15: hmc_core<15>(chain_idx, K, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A_tg); break;
        case 16: hmc_core<16>(chain_idx, K, L, eps, base_counter, thread_seed, q_in, q_out, accept_cnt, A_tg); break;
    }
}