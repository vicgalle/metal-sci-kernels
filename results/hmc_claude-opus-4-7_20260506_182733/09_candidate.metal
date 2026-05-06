#include <metal_stdlib>
using namespace metal;

inline uint mix32(uint x) {
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    x = x ^ (x >> 16);
    return x;
}

inline uint rand_u32(uint seed, uint chain_idx, uint counter) {
    uint x = seed + chain_idx * 0x9E3779B9u;
    x = mix32(x ^ counter);
    x = mix32(x);
    return x;
}

inline float u01_from_bits(uint bits) {
    return float(bits >> 8) * (1.0f / 16777216.0f);
}

// ============================================================================
// Templated worker: D is compile-time. Uses float4 loads from threadgroup A.
// Avoids storing q_old (re-reads q_in on reject) to reduce register pressure.
// ============================================================================
template <uint D>
inline void hmc_run(uint chain_idx,
                    device const float *q_in,
                    device       float *q_out,
                    device       uint  *accept_cnt,
                    threadgroup const float4 *Atile4, // row stride = D/4 float4s
                    uint L, float eps,
                    uint hmc_step_idx, uint seed)
{
    constexpr uint D4 = D / 4u;

    float4 q4[D4];
    float4 p4[D4];
    float4 f4[D4];

    uint base_counter = hmc_step_idx * (D + 1u);
    float eps_half = 0.5f * eps;

    // 1) Momentum p ~ N(0, I) via Box-Muller, paired counters.
    #pragma unroll
    for (uint i = 0u; i < D; i += 2u) {
        uint b1 = rand_u32(seed, chain_idx, base_counter + i);
        uint b2 = rand_u32(seed, chain_idx, base_counter + i + 1u);
        float u1 = u01_from_bits(b1);
        float u2 = u01_from_bits(b2);
        u1 = max(u1, 1.0e-7f);
        float r = sqrt(-2.0f * log(u1));
        float angle = 6.2831853071795864f * u2;
        float c;
        float s = sincos(angle, c);
        // pack into float4 lanes
        uint vi = i >> 2;
        uint li = i & 3u;
        if (li == 0u) { p4[vi].x = r * c; p4[vi].y = r * s; }
        else          { p4[vi].z = r * c; p4[vi].w = r * s; }
    }

    // 2) Load q (and remember source pointer for reject path).
    device const float4 *qin4 = (device const float4 *)(q_in + chain_idx * D);
    #pragma unroll
    for (uint v = 0u; v < D4; ++v) q4[v] = qin4[v];

    // matvec f = A q  (row-by-row, vectorized inner product)
    #pragma unroll
    for (uint i = 0u; i < D; ++i) {
        threadgroup const float4 *Arow4 = Atile4 + i * D4;
        float4 acc = float4(0.0f);
        #pragma unroll
        for (uint v = 0u; v < D4; ++v) {
            acc = fma(Arow4[v], q4[v], acc);
        }
        float a = acc.x + acc.y + acc.z + acc.w;
        uint vi = i >> 2;
        uint li = i & 3u;
        if      (li == 0u) f4[vi].x = a;
        else if (li == 1u) f4[vi].y = a;
        else if (li == 2u) f4[vi].z = a;
        else               f4[vi].w = a;
    }

    // U_old, K_old
    float4 Uacc = float4(0.0f);
    float4 Kacc = float4(0.0f);
    #pragma unroll
    for (uint v = 0u; v < D4; ++v) {
        Uacc = fma(q4[v], f4[v], Uacc);
        Kacc = fma(p4[v], p4[v], Kacc);
    }
    float U_old = 0.5f * (Uacc.x + Uacc.y + Uacc.z + Uacc.w);
    float K_old = 0.5f * (Kacc.x + Kacc.y + Kacc.z + Kacc.w);

    // 3) Leapfrog: initial half-kick
    #pragma unroll
    for (uint v = 0u; v < D4; ++v) p4[v] = fma(float4(-eps_half), f4[v], p4[v]);

    for (uint l = 0u; l < L; ++l) {
        // drift
        #pragma unroll
        for (uint v = 0u; v < D4; ++v) q4[v] = fma(float4(eps), p4[v], q4[v]);
        // recompute force
        #pragma unroll
        for (uint i = 0u; i < D; ++i) {
            threadgroup const float4 *Arow4 = Atile4 + i * D4;
            float4 acc = float4(0.0f);
            #pragma unroll
            for (uint v = 0u; v < D4; ++v) {
                acc = fma(Arow4[v], q4[v], acc);
            }
            float a = acc.x + acc.y + acc.z + acc.w;
            uint vi = i >> 2;
            uint li = i & 3u;
            if      (li == 0u) f4[vi].x = a;
            else if (li == 1u) f4[vi].y = a;
            else if (li == 2u) f4[vi].z = a;
            else               f4[vi].w = a;
        }
        // kick
        float scale = (l + 1u == L) ? -eps_half : -eps;
        #pragma unroll
        for (uint v = 0u; v < D4; ++v) p4[v] = fma(float4(scale), f4[v], p4[v]);
    }

    // 4) U_new, K_new (reuse final force)
    float4 Uacc2 = float4(0.0f);
    float4 Kacc2 = float4(0.0f);
    #pragma unroll
    for (uint v = 0u; v < D4; ++v) {
        Uacc2 = fma(q4[v], f4[v], Uacc2);
        Kacc2 = fma(p4[v], p4[v], Kacc2);
    }
    float U_new = 0.5f * (Uacc2.x + Uacc2.y + Uacc2.z + Uacc2.w);
    float K_new = 0.5f * (Kacc2.x + Kacc2.y + Kacc2.z + Kacc2.w);

    // 5) Accept/reject
    float dH = (U_new + K_new) - (U_old + K_old);
    uint b_acc = rand_u32(seed, chain_idx, base_counter + D);
    float u_acc = u01_from_bits(b_acc);
    float log_u = log(max(u_acc, 1.0e-30f));
    bool accept = isfinite(dH) && (log_u < -dH);

    device float4 *qout4 = (device float4 *)(q_out + chain_idx * D);
    if (accept) {
        #pragma unroll
        for (uint v = 0u; v < D4; ++v) qout4[v] = q4[v];
        accept_cnt[chain_idx] = accept_cnt[chain_idx] + 1u;
    } else {
        // Re-read q_in to write q_old (saves D registers vs caching qold).
        #pragma unroll
        for (uint v = 0u; v < D4; ++v) qout4[v] = qin4[v];
    }
}

// ============================================================================
// Entry kernel: cooperatively cache A into threadgroup as float4, dispatch by d.
// ============================================================================
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
    // 32x32 = 1024 floats = 256 float4 = 4 KB.
    threadgroup float4 Atile4[256];
    threadgroup float *Atile = (threadgroup float *)Atile4;

    uint dd = d * d;
    for (uint k = tid_in_tg; k < dd; k += tg_size) {
        Atile[k] = A[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (chain_idx >= K) return;

    if (d == 8u) {
        hmc_run<8u>(chain_idx, q_in, q_out, accept_cnt, Atile4, L, eps, hmc_step_idx, seed);
    } else if (d == 16u) {
        hmc_run<16u>(chain_idx, q_in, q_out, accept_cnt, Atile4, L, eps, hmc_step_idx, seed);
    } else {
        hmc_run<32u>(chain_idx, q_in, q_out, accept_cnt, Atile4, L, eps, hmc_step_idx, seed);
    }
}