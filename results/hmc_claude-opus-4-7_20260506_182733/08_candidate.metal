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
// Templated worker: D is compile-time; uses float4 loads from threadgroup A.
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

    float q[D];
    float p[D];
    float f[D];

    uint base_counter = hmc_step_idx * (D + 1u);
    float eps_half = 0.5f * eps;

    // 1) Momentum p ~ N(0, I)
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
        p[i] = r * c;
        p[i + 1u] = r * s;
    }

    // 2) Load q
    device const float *qin = q_in + chain_idx * D;
    #pragma unroll
    for (uint i = 0u; i < D; ++i) q[i] = qin[i];

    // matvec f = A q with float4 loads from threadgroup
    #pragma unroll
    for (uint i = 0u; i < D; ++i) {
        threadgroup const float4 *Arow4 = Atile4 + i * D4;
        float4 acc4 = float4(0.0f);
        #pragma unroll
        for (uint j4 = 0u; j4 < D4; ++j4) {
            float4 a = Arow4[j4];
            float4 qv = float4(q[j4*4u], q[j4*4u+1u], q[j4*4u+2u], q[j4*4u+3u]);
            acc4 = fma(a, qv, acc4);
        }
        f[i] = acc4.x + acc4.y + acc4.z + acc4.w;
    }

    // U_old, K_old
    float U_old = 0.0f;
    float K_old = 0.0f;
    #pragma unroll
    for (uint i = 0u; i < D; ++i) {
        U_old = fma(q[i], f[i], U_old);
        K_old = fma(p[i], p[i], K_old);
    }
    U_old *= 0.5f;
    K_old *= 0.5f;

    // 3) Leapfrog
    // initial half-kick
    #pragma unroll
    for (uint i = 0u; i < D; ++i) p[i] = fma(-eps_half, f[i], p[i]);

    for (uint l = 0u; l < L; ++l) {
        // drift
        #pragma unroll
        for (uint i = 0u; i < D; ++i) q[i] = fma(eps, p[i], q[i]);
        // recompute force with float4 loads
        #pragma unroll
        for (uint i = 0u; i < D; ++i) {
            threadgroup const float4 *Arow4 = Atile4 + i * D4;
            float4 acc4 = float4(0.0f);
            #pragma unroll
            for (uint j4 = 0u; j4 < D4; ++j4) {
                float4 a = Arow4[j4];
                float4 qv = float4(q[j4*4u], q[j4*4u+1u], q[j4*4u+2u], q[j4*4u+3u]);
                acc4 = fma(a, qv, acc4);
            }
            f[i] = acc4.x + acc4.y + acc4.z + acc4.w;
        }
        // kick
        float scale = (l + 1u == L) ? -eps_half : -eps;
        #pragma unroll
        for (uint i = 0u; i < D; ++i) p[i] = fma(scale, f[i], p[i]);
    }

    // 4) U_new, K_new
    float U_new = 0.0f;
    float K_new = 0.0f;
    #pragma unroll
    for (uint i = 0u; i < D; ++i) {
        U_new = fma(q[i], f[i], U_new);
        K_new = fma(p[i], p[i], K_new);
    }
    U_new *= 0.5f;
    K_new *= 0.5f;

    // 5) Accept/reject
    float dH = (U_new + K_new) - (U_old + K_old);
    uint b_acc = rand_u32(seed, chain_idx, base_counter + D);
    float u_acc = u01_from_bits(b_acc);
    float log_u = log(max(u_acc, 1.0e-30f));
    bool accept = isfinite(dH) && (log_u < -dH);

    device float *qout = q_out + chain_idx * D;
    if (accept) {
        #pragma unroll
        for (uint i = 0u; i < D; ++i) qout[i] = q[i];
        accept_cnt[chain_idx] = accept_cnt[chain_idx] + 1u;
    } else {
        // Reject: write q_old (== q_in)
        #pragma unroll
        for (uint i = 0u; i < D; ++i) qout[i] = qin[i];
    }
}

// ============================================================================
// Entry: cooperatively cache A into threadgroup as float4, then dispatch by d.
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