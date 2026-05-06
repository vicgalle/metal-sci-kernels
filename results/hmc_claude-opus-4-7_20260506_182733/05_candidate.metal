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

// Compute force = A * q using fully-unrolled float4 inner loop.
// Atile is laid out with row stride = D_MAX (32 floats = 8 float4).
inline void matvec(thread float4 *f4,
                   threadgroup const float4 *Atile4,
                   thread const float4 *q4,
                   uint d) {
    // Process two output rows at a time to reuse q4 loads.
    for (uint i = 0u; i < d; ++i) {
        threadgroup const float4 *Arow = Atile4 + i * D_MAX4;
        float4 acc = Arow[0] * q4[0];
        acc = fma(Arow[1], q4[1], acc);
        acc = fma(Arow[2], q4[2], acc);
        acc = fma(Arow[3], q4[3], acc);
        acc = fma(Arow[4], q4[4], acc);
        acc = fma(Arow[5], q4[5], acc);
        acc = fma(Arow[6], q4[6], acc);
        acc = fma(Arow[7], q4[7], acc);
        ((thread float *)f4)[i] = acc.x + acc.y + acc.z + acc.w;
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
                     uint tg_size   [[threads_per_threadgroup]]) {
    // Atile: D_MAX rows x D_MAX cols, row stride = D_MAX (32) floats.
    // 32 * 32 * 4 = 4096 bytes — well within threadgroup budget.
    threadgroup float4 Atile4[D_MAX * D_MAX4];

    // Cooperative zero-fill.
    threadgroup float *Atile = (threadgroup float *)Atile4;
    uint tile_sz = D_MAX * D_MAX;
    for (uint k = tid_in_tg; k < tile_sz; k += tg_size) {
        Atile[k] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load A[i,j] -> Atile[i*D_MAX + j] (zero-padded columns).
    uint dd = d * d;
    for (uint k = tid_in_tg; k < dd; k += tg_size) {
        uint i = k / d;
        uint j = k - i * d;
        Atile[i * D_MAX + j] = A[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (chain_idx >= K) return;

    // Register-resident state, all sized to D_MAX (=8 float4s).
    float4 q4[D_MAX4];
    float4 p4[D_MAX4];
    float4 f4[D_MAX4];
    float4 qold4[D_MAX4];

    #pragma unroll
    for (uint i = 0u; i < D_MAX4; ++i) {
        q4[i] = float4(0.0f);
        p4[i] = float4(0.0f);
        f4[i] = float4(0.0f);
        qold4[i] = float4(0.0f);
    }

    thread float *qs = (thread float *)q4;
    thread float *ps = (thread float *)p4;
    thread float *fs = (thread float *)f4;
    thread float *qos = (thread float *)qold4;

    uint base_counter = hmc_step_idx * (d + 1u);
    float eps_half = 0.5f * eps;

    // 1. Momentum p ~ N(0,I) via Box-Muller.
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
        ps[i] = r * c;
        if (i + 1u < d) ps[i + 1u] = r * s;
    }

    // 2. Load q, save q_old.
    device const float *qin = q_in + chain_idx * d;
    for (uint i = 0u; i < d; ++i) {
        float qi = qin[i];
        qs[i] = qi;
        qos[i] = qi;
    }

    // Initial force and U_old.
    matvec(f4, Atile4, q4, d);
    float U_old = 0.0f;
    for (uint i = 0u; i < d; ++i) U_old = fma(0.5f * qs[i], fs[i], U_old);

    float K_old = 0.0f;
    #pragma unroll
    for (uint j4 = 0u; j4 < D_MAX4; ++j4) K_old += dot(p4[j4], p4[j4]);
    K_old *= 0.5f;

    // 3. Leapfrog.
    // Initial half-kick.
    float4 nh4 = float4(-eps_half);
    float4 ne4 = float4(-eps);
    float4 e4  = float4(eps);

    #pragma unroll
    for (uint j4 = 0u; j4 < D_MAX4; ++j4) {
        p4[j4] = fma(nh4, f4[j4], p4[j4]);
    }

    if (L > 0u) {
        // l = 0 .. L-2 with full kick.
        for (uint l = 0u; l + 1u < L; ++l) {
            #pragma unroll
            for (uint j4 = 0u; j4 < D_MAX4; ++j4) {
                q4[j4] = fma(e4, p4[j4], q4[j4]);
            }
            matvec(f4, Atile4, q4, d);
            #pragma unroll
            for (uint j4 = 0u; j4 < D_MAX4; ++j4) {
                p4[j4] = fma(ne4, f4[j4], p4[j4]);
            }
        }
        // Last iter: half-kick.
        #pragma unroll
        for (uint j4 = 0u; j4 < D_MAX4; ++j4) {
            q4[j4] = fma(e4, p4[j4], q4[j4]);
        }
        matvec(f4, Atile4, q4, d);
        #pragma unroll
        for (uint j4 = 0u; j4 < D_MAX4; ++j4) {
            p4[j4] = fma(nh4, f4[j4], p4[j4]);
        }
    }

    // 4. New Hamiltonian.
    float U_new = 0.0f;
    for (uint i = 0u; i < d; ++i) U_new = fma(0.5f * qs[i], fs[i], U_new);
    float K_new = 0.0f;
    #pragma unroll
    for (uint j4 = 0u; j4 < D_MAX4; ++j4) K_new += dot(p4[j4], p4[j4]);
    K_new *= 0.5f;

    // 5. Accept/reject.
    float dH = (U_new + K_new) - (U_old + K_old);
    uint b_acc = rand_u32(seed, chain_idx, base_counter + d);
    float u_acc = u01_from_bits(b_acc);
    float log_u = log(max(u_acc, 1.0e-30f));
    bool accept = isfinite(dH) && (log_u < -dH);

    device float *qout = q_out + chain_idx * d;
    for (uint i = 0u; i < d; ++i) {
        qout[i] = accept ? qs[i] : qos[i];
    }
    if (accept) {
        accept_cnt[chain_idx] = accept_cnt[chain_idx] + 1u;
    }
}