#include <metal_stdlib>
using namespace metal;

#define D_MAX 32u
#define TG_SIZE 64u

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
    // Cooperative load of A (d*d <= 1024 floats = 4KB) into threadgroup mem.
    threadgroup float Atile[D_MAX * D_MAX];

    uint dd = d * d;
    for (uint k = tid_in_tg; k < dd; k += tg_size) {
        Atile[k] = A[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (chain_idx >= K) return;

    thread float q[D_MAX];
    thread float p[D_MAX];
    thread float q_old[D_MAX];
    thread float force[D_MAX];

    uint base_counter = hmc_step_idx * (d + 1u);
    float eps_half = 0.5f * eps;

    // 1. Momentum p ~ N(0, I), Box-Muller.
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
        if (i + 1u < d) {
            p[i + 1u] = r * s;
        }
    }

    // 2. Load q, save q_old.
    for (uint i = 0u; i < d; ++i) {
        float qi = q_in[chain_idx * d + i];
        q[i] = qi;
        q_old[i] = qi;
    }

    // Initial force = A q, U_old.
    float U_old = 0.0f;
    for (uint i = 0u; i < d; ++i) {
        float acc = 0.0f;
        threadgroup const float *Ar = &Atile[i * d];
        for (uint j = 0u; j < d; ++j) {
            acc = fma(Ar[j], q[j], acc);
        }
        force[i] = acc;
        U_old = fma(0.5f * q[i], acc, U_old);
    }
    float K_old = 0.0f;
    for (uint i = 0u; i < d; ++i) K_old = fma(0.5f * p[i], p[i], K_old);

    // 3. Leapfrog.
    for (uint i = 0u; i < d; ++i) p[i] -= eps_half * force[i];

    for (uint l = 0u; l < L; ++l) {
        for (uint i = 0u; i < d; ++i) q[i] = fma(eps, p[i], q[i]);
        for (uint i = 0u; i < d; ++i) {
            float acc = 0.0f;
            threadgroup const float *Ar = &Atile[i * d];
            for (uint j = 0u; j < d; ++j) {
                acc = fma(Ar[j], q[j], acc);
            }
            force[i] = acc;
        }
        float scale = (l + 1u == L) ? eps_half : eps;
        for (uint i = 0u; i < d; ++i) p[i] -= scale * force[i];
    }

    // 4. New Hamiltonian.
    float U_new = 0.0f;
    for (uint i = 0u; i < d; ++i) U_new = fma(0.5f * q[i], force[i], U_new);
    float K_new = 0.0f;
    for (uint i = 0u; i < d; ++i) K_new = fma(0.5f * p[i], p[i], K_new);

    // 5. Accept/reject.
    float dH = (U_new + K_new) - (U_old + K_old);
    uint b_acc = rand_u32(seed, chain_idx, base_counter + d);
    float u_acc = u01_from_bits(b_acc);
    float log_u = log(max(u_acc, 1.0e-30f));
    bool accept = isfinite(dH) && (log_u < -dH);

    for (uint i = 0u; i < d; ++i) {
        q_out[chain_idx * d + i] = accept ? q[i] : q_old[i];
    }
    if (accept) {
        accept_cnt[chain_idx] = accept_cnt[chain_idx] + 1u;
    }
}