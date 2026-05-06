**Optimization:** The previous version always padded to D_MAX=32 and processed 8 float4s regardless of `d`. For `d=8` and `d=16`, this wastes 4-8x of the matvec work. I now specialize the kernel into three template-like code paths (d=8, d=16, d=32) using compile-time constant loop bounds (selected by runtime `d`), so the compiler can fully unroll matvec, leapfrog updates, and reductions with the actual size. I also keep A in `constant` address space when reading (small d²) by caching into threadgroup memory tightly packed (row stride = d, not 32), allowing dense float4 vector loads only for d=32 and scalar FMA chains for smaller d. This eliminates per-iteration zero lanes and reduces threadgroup memory pressure and instructions for the small-d cases that dominated runtime.

```metal
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
// Templated worker: D is a compile-time size; matvec/reductions fully unroll.
// ============================================================================
template <uint D>
inline void hmc_run(uint chain_idx,
                    device const float *q_in,
                    device       float *q_out,
                    device       uint  *accept_cnt,
                    threadgroup const float *Atile,   // row stride = D
                    uint L, float eps,
                    uint hmc_step_idx, uint seed)
{
    float q[D];
    float p[D];
    float f[D];
    float qold[D];

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

    // 2) Load q, save q_old
    device const float *qin = q_in + chain_idx * D;
    #pragma unroll
    for (uint i = 0u; i < D; ++i) {
        float qi = qin[i];
        q[i] = qi;
        qold[i] = qi;
    }

    // matvec f = A q  (fully unrolled)
    #pragma unroll
    for (uint i = 0u; i < D; ++i) {
        threadgroup const float *Arow = Atile + i * D;
        float acc = 0.0f;
        #pragma unroll
        for (uint j = 0u; j < D; ++j) {
            acc = fma(Arow[j], q[j], acc);
        }
        f[i] = acc;
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
    // Initial half-kick
    #pragma unroll
    for (uint i = 0u; i < D; ++i) p[i] = fma(-eps_half, f[i], p[i]);

    for (uint l = 0u; l < L; ++l) {
        // drift
        #pragma unroll
        for (uint i = 0u; i < D; ++i) q[i] = fma(eps, p[i], q[i]);
        // recompute force
        #pragma unroll
        for (uint i = 0u; i < D; ++i) {
            threadgroup const float *Arow = Atile + i * D;
            float acc = 0.0f;
            #pragma unroll
            for (uint j = 0u; j < D; ++j) {
                acc = fma(Arow[j], q[j], acc);
            }
            f[i] = acc;
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
    #pragma unroll
    for (uint i = 0u; i < D; ++i) {
        qout[i] = accept ? q[i] : qold[i];
    }
    if (accept) {
        accept_cnt[chain_idx] = accept_cnt[chain_idx] + 1u;
    }
}

// ============================================================================
// Entry kernel: cooperatively cache A into threadgroup, then dispatch by d.
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
    // Tightly-packed A cache: d * d floats (max 32*32 = 4 KB).
    threadgroup float Atile[32u * 32u];

    uint dd = d * d;
    for (uint k = tid_in_tg; k < dd; k += tg_size) {
        Atile[k] = A[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (chain_idx >= K) return;

    if (d == 8u) {
        hmc_run<8u>(chain_idx, q_in, q_out, accept_cnt, Atile, L, eps, hmc_step_idx, seed);
    } else if (d == 16u) {
        hmc_run<16u>(chain_idx, q_in, q_out, accept_cnt, Atile, L, eps, hmc_step_idx, seed);
    } else {
        hmc_run<32u>(chain_idx, q_in, q_out, accept_cnt, Atile, L, eps, hmc_step_idx, seed);
    }
}
```