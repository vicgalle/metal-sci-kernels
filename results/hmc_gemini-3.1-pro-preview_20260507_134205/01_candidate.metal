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
    float2 q_old[D_HALF];
    float2 force[D_HALF];

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

    // 2. Load q, save q_old
    #pragma clang loop unroll(full)
    for (uint i = 0; i < D_HALF; ++i) {
        float2 qi = q_in_vec[chain_idx * D_HALF + i];
        q[i] = qi;
        q_old[i] = qi;
    }

    // 3. Compute U_old and initial force
    float U_old = 0.0f;
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
        force[i] = fi;
        U_old += 0.5f * dot(q[i], fi);
    }

    float K_old = 0.0f;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < D_HALF; ++i) {
        K_old += 0.5f * dot(p[i], p[i]);
    }

    // 4. Leapfrog
    #pragma clang loop unroll(full)
    for (uint i = 0; i < D_HALF; ++i) {
        p[i] -= 0.5f * eps * force[i];
    }

    for (uint l = 0; l < L; ++l) {
        // drift
        #pragma clang loop unroll(full)
        for (uint i = 0; i < D_HALF; ++i) {
            q[i] += eps * p[i];
        }
        // recompute force
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
            force[i] = float2(acc0, acc1);
        }
        // kick
        float scale = (l + 1u == L) ? (0.5f * eps) : eps;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < D_HALF; ++i) {
            p[i] -= scale * force[i];
        }
    }

    // 5. Compute U_new and K_new
    float U_new = 0.0f;
    float K_new = 0.0f;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < D_HALF; ++i) {
        U_new += 0.5f * dot(q[i], force[i]);
        K_new += 0.5f * dot(p[i], p[i]);
    }

    // 6. Accept logic
    float dH = (U_new + K_new) - (U_old + K_old);
    uint b_acc = mix32(mix32(thread_seed ^ (base_counter + d)));
    float u_acc = float(b_acc >> 8) * (1.0f / 16777216.0f);
    float log_u = log(max(u_acc, 1.0e-30f));
    bool accept = isfinite(dH) && (log_u < -dH);

    #pragma clang loop unroll(full)
    for (uint i = 0; i < D_HALF; ++i) {
        q_out_vec[chain_idx * D_HALF + i] = accept ? q[i] : q_old[i];
    }
    if (accept) {
        accept_cnt[chain_idx] += 1u;
    }
}

// Fallback for non-standard even sizes.
inline void hmc_core_dynamic(
    uint chain_idx, uint d, uint L, float eps, uint base_counter, uint thread_seed,
    device const float2* q_in_vec, device float2* q_out_vec, device uint* accept_cnt,
    threadgroup float2* A_shared)
{
    uint d_half = d / 2;
    float2 q[16];
    float2 p[16];
    float2 q_old[16];
    float2 force[16];

    for (uint i = 0; i < d_half; ++i) {
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

    for (uint i = 0; i < d_half; ++i) {
        float2 qi = q_in_vec[chain_idx * d_half + i];
        q[i] = qi;
        q_old[i] = qi;
    }

    float U_old = 0.0f;
    for (uint i = 0; i < d_half; ++i) {
        uint row0 = 2 * i;
        uint row1 = 2 * i + 1;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint j = 0; j < d_half; ++j) {
            float2 qj = q[j];
            acc0 += dot(A_shared[row0 * d_half + j], qj);
            acc1 += dot(A_shared[row1 * d_half + j], qj);
        }
        float2 fi = float2(acc0, acc1);
        force[i] = fi;
        U_old += 0.5f * dot(q[i], fi);
    }

    float K_old = 0.0f;
    for (uint i = 0; i < d_half; ++i) {
        K_old += 0.5f * dot(p[i], p[i]);
    }

    for (uint i = 0; i < d_half; ++i) {
        p[i] -= 0.5f * eps * force[i];
    }

    for (uint l = 0; l < L; ++l) {
        for (uint i = 0; i < d_half; ++i) {
            q[i] += eps * p[i];
        }
        for (uint i = 0; i < d_half; ++i) {
            uint row0 = 2 * i;
            uint row1 = 2 * i + 1;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            for (uint j = 0; j < d_half; ++j) {
                float2 qj = q[j];
                acc0 += dot(A_shared[row0 * d_half + j], qj);
                acc1 += dot(A_shared[row1 * d_half + j], qj);
            }
            force[i] = float2(acc0, acc1);
        }
        float scale = (l + 1u == L) ? (0.5f * eps) : eps;
        for (uint i = 0; i < d_half; ++i) {
            p[i] -= scale * force[i];
        }
    }

    float U_new = 0.0f;
    float K_new = 0.0f;
    for (uint i = 0; i < d_half; ++i) {
        U_new += 0.5f * dot(q[i], force[i]);
        K_new += 0.5f * dot(p[i], p[i]);
    }

    float dH = (U_new + K_new) - (U_old + K_old);
    uint b_acc = mix32(mix32(thread_seed ^ (base_counter + d)));
    float u_acc = float(b_acc >> 8) * (1.0f / 16777216.0f);
    float log_u = log(max(u_acc, 1.0e-30f));
    bool accept = isfinite(dH) && (log_u < -dH);

    for (uint i = 0; i < d_half; ++i) {
        q_out_vec[chain_idx * d_half + i] = accept ? q[i] : q_old[i];
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

    // Dispatch to strongly-typed sizes for aggressive unrolling.
    if (d_half == 4) {
        hmc_core<4>(chain_idx, d, L, eps, base_counter, thread_seed, q_in_vec, q_out_vec, accept_cnt, A_shared);
    } else if (d_half == 8) {
        hmc_core<8>(chain_idx, d, L, eps, base_counter, thread_seed, q_in_vec, q_out_vec, accept_cnt, A_shared);
    } else if (d_half == 16) {
        hmc_core<16>(chain_idx, d, L, eps, base_counter, thread_seed, q_in_vec, q_out_vec, accept_cnt, A_shared);
    } else {
        hmc_core_dynamic(chain_idx, d, L, eps, base_counter, thread_seed, q_in_vec, q_out_vec, accept_cnt, A_shared);
    }
}