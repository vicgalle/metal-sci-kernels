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