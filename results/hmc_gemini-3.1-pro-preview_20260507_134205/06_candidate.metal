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