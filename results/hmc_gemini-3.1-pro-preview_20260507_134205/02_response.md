The catastrophic performance cliff at $d=16$ and $d=32$ in the previous attempt was caused by **register spilling to device local memory**. Fully unrolling both the inner force computation loops and the outer leapfrog step resulted in massive code size, while keeping local arrays (`force`, `q_old`, `q`, `p`) caused register pressure to exceed the 128-register limit, forcing dynamic array indices to spill to memory.

This optimization implements a zero-spill, fully branchless leapfrog iteration:
1. **Eliminated `force` and `q_old` arrays**: By fusing the momentum kick into the force computation and dynamically checking `q_in` for rejected proposals, we cut local register array sizes by 50%.
2. **Exhaustive Template Instantiation**: Because $d \le 32$ and $d$ is even, we generate exactly 16 templates. This guarantees that `#pragma clang loop unroll(full)` is fully evaluated by the compiler to static register addresses.
3. **Loop Peeling for `U_new`**: Peeling the final $L$-loop iteration into its own unrolled block eliminates conditional checks within the inner leapfrog loop while cleanly harvesting the final force vector.

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