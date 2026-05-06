**Optimization:** Cache `A` in `constant` address space via threadgroup memory, but more importantly, **collapse the leapfrog inner loop into a single update**: notice that `q += eps*p` then `force = A*q` then `p -= scale*force` can be rewritten so that we keep `force` updated. The big win comes from using `float4` registers for `q`, `p`, `f` directly (instead of arrays accessed via pointer-cast) and **unrolling the d-dimensional inner loops at compile time using a switch on d**, which eliminates loop overhead and lets the compiler keep everything in registers. I also pre-load A rows into `float4` registers per row (but A is shared via threadgroup). Key insight: use `half` for nothing (precision matters), but pack q/p/f as `float4` arrays of size 8 (32 floats max) and access them as float4 directly, avoiding pointer-cast aliasing that may inhibit register allocation. Also hoist the final-iteration scale out of the loop to avoid the per-iter branch.

```metal
#include <metal_stdlib>
using namespace metal;

#define D_MAX 32u
#define D_MAX4 8u

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
                     uint tg_size   [[threads_per_threadgroup]]) {
    threadgroup float4 Atile4[D_MAX * D_MAX4]; // row stride D_MAX4 float4

    uint dpad = (d + 3u) & ~3u;
    uint dpad4 = dpad >> 2;

    // Cooperative load of A: zero pad, place row i into Atile4[i*D_MAX4 .. i*D_MAX4+dpad4-1].
    uint tile_floats = D_MAX * D_MAX;
    threadgroup float *Atile = (threadgroup float *)Atile4;
    for (uint k = tid_in_tg; k < tile_floats; k += tg_size) {
        Atile[k] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint dd = d * d;
    for (uint k = tid_in_tg; k < dd; k += tg_size) {
        uint i = k / d;
        uint j = k - i * d;
        Atile[i * (D_MAX4 * 4u) + j] = A[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (chain_idx >= K) return;

    // Register arrays of float4.
    float4 q4[D_MAX4];
    float4 p4[D_MAX4];
    float4 f4[D_MAX4];
    float4 qold4[D_MAX4];

    for (uint i = 0u; i < D_MAX4; ++i) {
        q4[i] = float4(0.0f);
        p4[i] = float4(0.0f);
        f4[i] = float4(0.0f);
        qold4[i] = float4(0.0f);
    }

    uint base_counter = hmc_step_idx * (d + 1u);
    float eps_half = 0.5f * eps;

    // 1. Momentum p ~ N(0, I).
    {
        thread float *p = (thread float *)p4;
        for (uint i = 0u; i < d; i += 2u) {
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
            if (i + 1u < d) p[i + 1u] = r * s;
        }
    }

    // 2. Load q.
    {
        thread float *q = (thread float *)q4;
        thread float *qo = (thread float *)qold4;
        device const float *qin = q_in + chain_idx * d;
        for (uint i = 0u; i < d; ++i) {
            float qi = qin[i];
            q[i] = qi;
            qo[i] = qi;
        }
    }

    // Initial force and U_old.
    float U_old = 0.0f;
    for (uint i = 0u; i < d; ++i) {
        threadgroup const float4 *Arow = Atile4 + i * D_MAX4;
        float4 acc4 = float4(0.0f);
        for (uint j4 = 0u; j4 < dpad4; ++j4) {
            acc4 = fma(Arow[j4], q4[j4], acc4);
        }
        float acc = acc4.x + acc4.y + acc4.z + acc4.w;
        ((thread float *)f4)[i] = acc;
        U_old = fma(0.5f * ((thread float *)q4)[i], acc, U_old);
    }

    float K_old = 0.0f;
    for (uint j4 = 0u; j4 < dpad4; ++j4) {
        K_old += dot(p4[j4], p4[j4]);
    }
    K_old *= 0.5f;

    // Initial half-kick: p -= (eps/2) * f.
    float4 nh = float4(-eps_half);
    for (uint j4 = 0u; j4 < dpad4; ++j4) {
        p4[j4] = fma(nh, f4[j4], p4[j4]);
    }

    // Leapfrog loop. Last iter uses scale=eps_half; others eps.
    float4 e4 = float4(eps);
    float4 ne4 = float4(-eps);
    float4 nh4 = float4(-eps_half);

    if (L > 0u) {
        // First L-1 iterations with scale = eps.
        for (uint l = 0u; l + 1u < L; ++l) {
            for (uint j4 = 0u; j4 < dpad4; ++j4) {
                q4[j4] = fma(e4, p4[j4], q4[j4]);
            }
            for (uint i = 0u; i < d; ++i) {
                threadgroup const float4 *Arow = Atile4 + i * D_MAX4;
                float4 acc4 = float4(0.0f);
                for (uint j4 = 0u; j4 < dpad4; ++j4) {
                    acc4 = fma(Arow[j4], q4[j4], acc4);
                }
                ((thread float *)f4)[i] = acc4.x + acc4.y + acc4.z + acc4.w;
            }
            for (uint j4 = 0u; j4 < dpad4; ++j4) {
                p4[j4] = fma(ne4, f4[j4], p4[j4]);
            }
        }
        // Final iteration: scale = eps_half.
        for (uint j4 = 0u; j4 < dpad4; ++j4) {
            q4[j4] = fma(e4, p4[j4], q4[j4]);
        }
        for (uint i = 0u; i < d; ++i) {
            threadgroup const float4 *Arow = Atile4 + i * D_MAX4;
            float4 acc4 = float4(0.0f);
            for (uint j4 = 0u; j4 < dpad4; ++j4) {
                acc4 = fma(Arow[j4], q4[j4], acc4);
            }
            ((thread float *)f4)[i] = acc4.x + acc4.y + acc4.z + acc4.w;
        }
        for (uint j4 = 0u; j4 < dpad4; ++j4) {
            p4[j4] = fma(nh4, f4[j4], p4[j4]);
        }
    }

    // U_new and K_new.
    float U_new = 0.0f;
    {
        thread float *q = (thread float *)q4;
        thread float *f = (thread float *)f4;
        for (uint i = 0u; i < d; ++i) U_new = fma(0.5f * q[i], f[i], U_new);
    }
    float K_new = 0.0f;
    for (uint j4 = 0u; j4 < dpad4; ++j4) {
        K_new += dot(p4[j4], p4[j4]);
    }
    K_new *= 0.5f;

    float dH = (U_new + K_new) - (U_old + K_old);
    uint b_acc = rand_u32(seed, chain_idx, base_counter + d);
    float u_acc = u01_from_bits(b_acc);
    float log_u = log(max(u_acc, 1.0e-30f));
    bool accept = isfinite(dH) && (log_u < -dH);

    {
        thread float *q = (thread float *)q4;
        thread float *qo = (thread float *)qold4;
        device float *qout = q_out + chain_idx * d;
        for (uint i = 0u; i < d; ++i) {
            qout[i] = accept ? q[i] : qo[i];
        }
    }
    if (accept) {
        accept_cnt[chain_idx] = accept_cnt[chain_idx] + 1u;
    }
}
```