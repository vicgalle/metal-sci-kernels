I replace scalar threadgroup “float4” construction with a true `threadgroup float4` transposed A layout, reducing address arithmetic and enabling vector loads. I also specialize d=16/d=32 matvecs to compute 16 output rows per pass, cutting repeated q reads and loop overhead while preserving one thread per chain and the same RNG/counter behavior. The prior compile issue is avoided by not using the misplaced function attribute and by fixing the dynamic drift call.

```metal
#include <metal_stdlib>
using namespace metal;

#define D_MAX 32u

inline uint mix32(uint x) {
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    return x ^ (x >> 16);
}

inline uint rand_u32_chain(uint chain_seed, uint counter) {
    uint x = mix32(chain_seed ^ counter);
    return mix32(x);
}

inline float u01_from_bits(uint bits) {
    return float(bits >> 8) * (1.0f / 16777216.0f);
}

inline void load_A_transpose4_tg(device const float *A,
                                 threadgroup float4 *AT4,
                                 uint d,
                                 uint tid,
                                 uint tpg) {
    uint blocks = d >> 2;
    uint nvec = d * blocks;

    for (uint idx = tid; idx < nvec; idx += tpg) {
        uint col = idx / blocks;
        uint b = idx - col * blocks;
        uint row = b << 2;
        AT4[idx] = float4(A[(row + 0u) * d + col],
                          A[(row + 1u) * d + col],
                          A[(row + 2u) * d + col],
                          A[(row + 3u) * d + col]);
    }
}

inline bool accept_hmc_chain(float dH, uint chain_seed, uint acc_counter) {
    if (!isfinite(dH)) {
        return false;
    }
    if (dH <= 0.0f) {
        return true;
    }

    uint b_acc = rand_u32_chain(chain_seed, acc_counter);
    float u_acc = u01_from_bits(b_acc);
    float log_u = log(max(u_acc, 1.0e-30f));
    return log_u < -dH;
}

inline float half_dot4_ordered(float4 q, float4 f) {
    float u = 0.0f;
    u += 0.5f * q.x * f.x;
    u += 0.5f * q.y * f.y;
    u += 0.5f * q.z * f.z;
    u += 0.5f * q.w * f.w;
    return u;
}

template<uint D, bool COMPUTE_U>
inline float matvec_kick_chunk8(threadgroup const float4 *AT4,
                                thread const float4 *q,
                                thread float4 *p,
                                float scale) {
    constexpr uint B = D / 4u;
    float U = 0.0f;

#pragma unroll
    for (uint r = 0u; r < D; r += 8u) {
        uint rb = r >> 2;
        float4 acc0 = float4(0.0f);
        float4 acc1 = float4(0.0f);

#pragma unroll
        for (uint v = 0u; v < B; ++v) {
            float4 qv = q[v];
            uint j = v << 2;

            threadgroup const float4 *c0 = AT4 + (j + 0u) * B + rb;
            threadgroup const float4 *c1 = AT4 + (j + 1u) * B + rb;
            threadgroup const float4 *c2 = AT4 + (j + 2u) * B + rb;
            threadgroup const float4 *c3 = AT4 + (j + 3u) * B + rb;

            acc0 += c0[0] * qv.x;
            acc1 += c0[1] * qv.x;

            acc0 += c1[0] * qv.y;
            acc1 += c1[1] * qv.y;

            acc0 += c2[0] * qv.z;
            acc1 += c2[1] * qv.z;

            acc0 += c3[0] * qv.w;
            acc1 += c3[1] * qv.w;
        }

        if (COMPUTE_U) {
            U += half_dot4_ordered(q[rb],      acc0);
            U += half_dot4_ordered(q[rb + 1u], acc1);
        }

        p[rb]      -= scale * acc0;
        p[rb + 1u] -= scale * acc1;
    }

    return U;
}

template<uint D, bool COMPUTE_U>
inline float matvec_kick_chunk16(threadgroup const float4 *AT4,
                                 thread const float4 *q,
                                 thread float4 *p,
                                 float scale) {
    constexpr uint B = D / 4u;
    float U = 0.0f;

#pragma unroll
    for (uint r = 0u; r < D; r += 16u) {
        uint rb = r >> 2;
        float4 acc0 = float4(0.0f);
        float4 acc1 = float4(0.0f);
        float4 acc2 = float4(0.0f);
        float4 acc3 = float4(0.0f);

#pragma unroll
        for (uint v = 0u; v < B; ++v) {
            float4 qv = q[v];
            uint j = v << 2;

            threadgroup const float4 *c0 = AT4 + (j + 0u) * B + rb;
            threadgroup const float4 *c1 = AT4 + (j + 1u) * B + rb;
            threadgroup const float4 *c2 = AT4 + (j + 2u) * B + rb;
            threadgroup const float4 *c3 = AT4 + (j + 3u) * B + rb;

            acc0 += c0[0] * qv.x;
            acc1 += c0[1] * qv.x;
            acc2 += c0[2] * qv.x;
            acc3 += c0[3] * qv.x;

            acc0 += c1[0] * qv.y;
            acc1 += c1[1] * qv.y;
            acc2 += c1[2] * qv.y;
            acc3 += c1[3] * qv.y;

            acc0 += c2[0] * qv.z;
            acc1 += c2[1] * qv.z;
            acc2 += c2[2] * qv.z;
            acc3 += c2[3] * qv.z;

            acc0 += c3[0] * qv.w;
            acc1 += c3[1] * qv.w;
            acc2 += c3[2] * qv.w;
            acc3 += c3[3] * qv.w;
        }

        if (COMPUTE_U) {
            U += half_dot4_ordered(q[rb],      acc0);
            U += half_dot4_ordered(q[rb + 1u], acc1);
            U += half_dot4_ordered(q[rb + 2u], acc2);
            U += half_dot4_ordered(q[rb + 3u], acc3);
        }

        p[rb]      -= scale * acc0;
        p[rb + 1u] -= scale * acc1;
        p[rb + 2u] -= scale * acc2;
        p[rb + 3u] -= scale * acc3;
    }

    return U;
}

template<uint D>
inline void drift_fixed(thread float4 *q, thread const float4 *p, float eps) {
#pragma unroll
    for (uint v = 0u; v < (D / 4u); ++v) {
        q[v] += eps * p[v];
    }
}

template<uint D>
inline float kinetic_fixed(thread const float4 *p) {
    float Ksum = 0.0f;

#pragma unroll
    for (uint v = 0u; v < (D / 4u); ++v) {
        float4 pv = p[v];
        Ksum += 0.5f * pv.x * pv.x;
        Ksum += 0.5f * pv.y * pv.y;
        Ksum += 0.5f * pv.z * pv.z;
        Ksum += 0.5f * pv.w * pv.w;
    }

    return Ksum;
}

template<uint D>
inline void store_accept_fixed(device const float *qin,
                               device float *qout,
                               thread const float4 *q,
                               bool accept) {
    if (accept) {
#pragma unroll
        for (uint v = 0u; v < (D / 4u); ++v) {
            float4 qv = q[v];
            uint o = v << 2;
            qout[o + 0u] = qv.x;
            qout[o + 1u] = qv.y;
            qout[o + 2u] = qv.z;
            qout[o + 3u] = qv.w;
        }
    } else {
#pragma unroll
        for (uint v = 0u; v < (D / 4u); ++v) {
            uint o = v << 2;
            qout[o + 0u] = qin[o + 0u];
            qout[o + 1u] = qin[o + 1u];
            qout[o + 2u] = qin[o + 2u];
            qout[o + 3u] = qin[o + 3u];
        }
    }
}

template<uint D>
inline void init_fixed_state(device const float *q_in,
                             thread float4 *q,
                             thread float4 *p,
                             thread float &K_old,
                             uint base_counter,
                             uint chain_seed,
                             uint base) {
    K_old = 0.0f;

#pragma unroll
    for (uint v = 0u; v < (D / 4u); ++v) {
        uint i = v << 2;

        uint b1 = rand_u32_chain(chain_seed, base_counter + i);
        uint b2 = rand_u32_chain(chain_seed, base_counter + i + 1u);
        float u1 = max(u01_from_bits(b1), 1.0e-7f);
        float u2 = u01_from_bits(b2);
        float r = sqrt(-2.0f * log(u1));
        float angle = 6.2831853071795864f * u2;
        float c, s;
        s = sincos(angle, c);
        float p0 = r * c;
        float p1 = r * s;

        uint b3 = rand_u32_chain(chain_seed, base_counter + i + 2u);
        uint b4 = rand_u32_chain(chain_seed, base_counter + i + 3u);
        float u3 = max(u01_from_bits(b3), 1.0e-7f);
        float u4 = u01_from_bits(b4);
        float r2 = sqrt(-2.0f * log(u3));
        float angle2 = 6.2831853071795864f * u4;
        float c2, s2;
        s2 = sincos(angle2, c2);
        float p2 = r2 * c2;
        float p3 = r2 * s2;

        p[v] = float4(p0, p1, p2, p3);

        K_old += 0.5f * p0 * p0;
        K_old += 0.5f * p1 * p1;
        K_old += 0.5f * p2 * p2;
        K_old += 0.5f * p3 * p3;
    }

    device const float *qin = q_in + base;

#pragma unroll
    for (uint v = 0u; v < (D / 4u); ++v) {
        uint o = v << 2;
        q[v] = float4(qin[o + 0u], qin[o + 1u], qin[o + 2u], qin[o + 3u]);
    }
}

template<uint D>
inline void run_hmc_fixed8(device const float *q_in,
                           device float *q_out,
                           device uint *accept_cnt,
                           threadgroup const float4 *AT4,
                           uint L,
                           float eps,
                           uint hmc_step_idx,
                           uint seed,
                           uint chain_idx) {
    thread float4 q[D / 4u];
    thread float4 p[D / 4u];

    uint chain_seed = seed + chain_idx * 0x9E3779B9u;
    uint base_counter = hmc_step_idx * (D + 1u);
    uint base = chain_idx * D;

    float K_old;
    init_fixed_state<D>(q_in, q, p, K_old, base_counter, chain_seed, base);

    float half_eps = 0.5f * eps;
    float U_old = matvec_kick_chunk8<D, true>(AT4, q, p, half_eps);

    float U_new = U_old;
    if (L != 0u) {
        uint full_steps = L - 1u;

        for (uint l = 0u; l < full_steps; ++l) {
            drift_fixed<D>(q, p, eps);
            (void)matvec_kick_chunk8<D, false>(AT4, q, p, eps);
        }

        drift_fixed<D>(q, p, eps);
        U_new = matvec_kick_chunk8<D, true>(AT4, q, p, half_eps);
    }

    float K_new = kinetic_fixed<D>(p);
    float dH = (U_new + K_new) - (U_old + K_old);

    bool accept = accept_hmc_chain(dH, chain_seed, base_counter + D);

    device const float *qin = q_in + base;
    device float *qout = q_out + base;
    store_accept_fixed<D>(qin, qout, q, accept);

    if (accept) {
        accept_cnt[chain_idx] = accept_cnt[chain_idx] + 1u;
    }
}

template<uint D>
inline void run_hmc_fixed16(device const float *q_in,
                            device float *q_out,
                            device uint *accept_cnt,
                            threadgroup const float4 *AT4,
                            uint L,
                            float eps,
                            uint hmc_step_idx,
                            uint seed,
                            uint chain_idx) {
    thread float4 q[D / 4u];
    thread float4 p[D / 4u];

    uint chain_seed = seed + chain_idx * 0x9E3779B9u;
    uint base_counter = hmc_step_idx * (D + 1u);
    uint base = chain_idx * D;

    float K_old;
    init_fixed_state<D>(q_in, q, p, K_old, base_counter, chain_seed, base);

    float half_eps = 0.5f * eps;
    float U_old = matvec_kick_chunk16<D, true>(AT4, q, p, half_eps);

    float U_new = U_old;
    if (L != 0u) {
        uint full_steps = L - 1u;

        for (uint l = 0u; l < full_steps; ++l) {
            drift_fixed<D>(q, p, eps);
            (void)matvec_kick_chunk16<D, false>(AT4, q, p, eps);
        }

        drift_fixed<D>(q, p, eps);
        U_new = matvec_kick_chunk16<D, true>(AT4, q, p, half_eps);
    }

    float K_new = kinetic_fixed<D>(p);
    float dH = (U_new + K_new) - (U_old + K_old);

    bool accept = accept_hmc_chain(dH, chain_seed, base_counter + D);

    device const float *qin = q_in + base;
    device float *qout = q_out + base;
    store_accept_fixed<D>(qin, qout, q, accept);

    if (accept) {
        accept_cnt[chain_idx] = accept_cnt[chain_idx] + 1u;
    }
}

inline float matvec_kick_dynamic(device const float *A,
                                 thread const float *q,
                                 thread float *p,
                                 uint d,
                                 float scale,
                                 bool computeU) {
    float U = 0.0f;

    for (uint i = 0u; i < d; ++i) {
        float acc = 0.0f;
        device const float *row = A + i * d;

        for (uint j = 0u; j < d; ++j) {
            acc += row[j] * q[j];
        }

        if (computeU) {
            U += 0.5f * q[i] * acc;
        }

        p[i] -= scale * acc;
    }

    return U;
}

inline void drift_dynamic(thread float *q, thread const float *p, uint d, float eps) {
    for (uint i = 0u; i < d; ++i) {
        q[i] += eps * p[i];
    }
}

inline float kinetic_dynamic(thread const float *p, uint d) {
    float Ksum = 0.0f;

    for (uint i = 0u; i < d; ++i) {
        Ksum += 0.5f * p[i] * p[i];
    }

    return Ksum;
}

inline void run_hmc_dynamic(device const float *q_in,
                            device float *q_out,
                            device uint *accept_cnt,
                            device const float *A,
                            uint d,
                            uint L,
                            float eps,
                            uint hmc_step_idx,
                            uint seed,
                            uint chain_idx) {
    thread float q[D_MAX];
    thread float p[D_MAX];

    uint chain_seed = seed + chain_idx * 0x9E3779B9u;
    uint base_counter = hmc_step_idx * (d + 1u);
    uint base = chain_idx * d;

    float K_old = 0.0f;

    for (uint i = 0u; i < d; i += 2u) {
        uint b1 = rand_u32_chain(chain_seed, base_counter + i);
        uint b2 = rand_u32_chain(chain_seed, base_counter + i + 1u);

        float u1 = max(u01_from_bits(b1), 1.0e-7f);
        float u2 = u01_from_bits(b2);

        float r = sqrt(-2.0f * log(u1));
        float angle = 6.2831853071795864f * u2;

        float c, s;
        s = sincos(angle, c);

        float p0 = r * c;
        float p1 = r * s;

        p[i] = p0;
        p[i + 1u] = p1;

        K_old += 0.5f * p0 * p0;
        K_old += 0.5f * p1 * p1;
    }

    device const float *qin = q_in + base;

    for (uint i = 0u; i < d; ++i) {
        q[i] = qin[i];
    }

    float half_eps = 0.5f * eps;
    float U_old = matvec_kick_dynamic(A, q, p, d, half_eps, true);

    float U_new = U_old;
    if (L != 0u) {
        uint full_steps = L - 1u;

        for (uint l = 0u; l < full_steps; ++l) {
            drift_dynamic(q, p, d, eps);
            (void)matvec_kick_dynamic(A, q, p, d, eps, false);
        }

        drift_dynamic(q, p, d, eps);
        U_new = matvec_kick_dynamic(A, q, p, d, half_eps, true);
    }

    float K_new = kinetic_dynamic(p, d);
    float dH = (U_new + K_new) - (U_old + K_old);

    bool accept = accept_hmc_chain(dH, chain_seed, base_counter + d);

    device float *qout = q_out + base;

    if (accept) {
        for (uint i = 0u; i < d; ++i) {
            qout[i] = q[i];
        }
        accept_cnt[chain_idx] = accept_cnt[chain_idx] + 1u;
    } else {
        for (uint i = 0u; i < d; ++i) {
            qout[i] = qin[i];
        }
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
                     uint tid       [[thread_index_in_threadgroup]],
                     uint tpg       [[threads_per_threadgroup]]) {
    threadgroup float4 AT4[256];

    bool fixed_d = (d == 8u) || (d == 16u) || (d == 24u) || (d == 32u);

    if (fixed_d) {
        load_A_transpose4_tg(A, AT4, d, tid, tpg);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (chain_idx >= K) {
        return;
    }

    if (d == 8u) {
        run_hmc_fixed8<8u>(q_in, q_out, accept_cnt, AT4, L, eps, hmc_step_idx, seed, chain_idx);
    } else if (d == 16u) {
        run_hmc_fixed16<16u>(q_in, q_out, accept_cnt, AT4, L, eps, hmc_step_idx, seed, chain_idx);
    } else if (d == 24u) {
        run_hmc_fixed8<24u>(q_in, q_out, accept_cnt, AT4, L, eps, hmc_step_idx, seed, chain_idx);
    } else if (d == 32u) {
        run_hmc_fixed16<32u>(q_in, q_out, accept_cnt, AT4, L, eps, hmc_step_idx, seed, chain_idx);
    } else {
        run_hmc_dynamic(q_in, q_out, accept_cnt, A, d, L, eps, hmc_step_idx, seed, chain_idx);
    }
}
```