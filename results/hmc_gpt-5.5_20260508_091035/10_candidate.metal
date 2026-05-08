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

inline float4 load4_tg(threadgroup const float *p) {
    return float4(p[0], p[1], p[2], p[3]);
}

// For d=16 store A row-major for a lower-register row-wise matvec.
// For all other fixed paths store A transposed as in the incumbent.
inline void load_A_mixed_tg(device const float *A,
                            threadgroup float *AT,
                            uint d,
                            uint tid,
                            uint tpg) {
    if (d == 8u) {
        for (uint idx = tid; idx < 64u; idx += tpg) {
            uint row = idx >> 3;
            uint col = idx & 7u;
            AT[(col << 3) + row] = A[idx];
        }
    } else if (d == 16u) {
        for (uint idx = tid; idx < 256u; idx += tpg) {
            AT[idx] = A[idx];
        }
    } else if (d == 24u) {
        for (uint idx = tid; idx < 576u; idx += tpg) {
            uint row = idx / 24u;
            uint col = idx - row * 24u;
            AT[col * 24u + row] = A[idx];
        }
    } else if (d == 32u) {
        for (uint idx = tid; idx < 1024u; idx += tpg) {
            uint row = idx >> 5;
            uint col = idx & 31u;
            AT[(col << 5) + row] = A[idx];
        }
    } else {
        uint n = d * d;
        for (uint idx = tid; idx < n; idx += tpg) {
            uint row = idx / d;
            uint col = idx - row * d;
            AT[col * d + row] = A[idx];
        }
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

template<uint D>
inline void init_fixed_state(device const float *qin,
                             thread float4 *q,
                             thread float4 *p,
                             thread float &K_old,
                             uint base_counter,
                             uint chain_seed) {
    device const float4 *qin4 = reinterpret_cast<device const float4 *>(qin);

#pragma unroll
    for (uint v = 0u; v < (D / 4u); ++v) {
        q[v] = qin4[v];
    }

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
}

template<uint D, bool COMPUTE_U>
inline float matvec_kick_chunk8(threadgroup const float *AT,
                                thread const float4 *q,
                                thread float4 *p,
                                float scale) {
    float U = 0.0f;

#pragma unroll
    for (uint r = 0u; r < D; r += 8u) {
        float4 acc0 = float4(0.0f);
        float4 acc1 = float4(0.0f);

#pragma unroll
        for (uint v = 0u; v < (D / 4u); ++v) {
            float4 qv = q[v];
            uint j = v << 2;

            threadgroup const float *c0 = AT + (j + 0u) * D + r;
            threadgroup const float *c1 = AT + (j + 1u) * D + r;
            threadgroup const float *c2 = AT + (j + 2u) * D + r;
            threadgroup const float *c3 = AT + (j + 3u) * D + r;

            acc0 += load4_tg(c0)      * qv.x;
            acc1 += load4_tg(c0 + 4u) * qv.x;

            acc0 += load4_tg(c1)      * qv.y;
            acc1 += load4_tg(c1 + 4u) * qv.y;

            acc0 += load4_tg(c2)      * qv.z;
            acc1 += load4_tg(c2 + 4u) * qv.z;

            acc0 += load4_tg(c3)      * qv.w;
            acc1 += load4_tg(c3 + 4u) * qv.w;
        }

        uint rv = r >> 2;

        if (COMPUTE_U) {
            float4 q0 = q[rv];
            float4 q1 = q[rv + 1u];

            U += 0.5f * q0.x * acc0.x;
            U += 0.5f * q0.y * acc0.y;
            U += 0.5f * q0.z * acc0.z;
            U += 0.5f * q0.w * acc0.w;

            U += 0.5f * q1.x * acc1.x;
            U += 0.5f * q1.y * acc1.y;
            U += 0.5f * q1.z * acc1.z;
            U += 0.5f * q1.w * acc1.w;
        }

        p[rv]      -= scale * acc0;
        p[rv + 1u] -= scale * acc1;
    }

    return U;
}

inline float mad4_order(float4 a, float4 b, float acc) {
    acc = fma(a.x, b.x, acc);
    acc = fma(a.y, b.y, acc);
    acc = fma(a.z, b.z, acc);
    acc = fma(a.w, b.w, acc);
    return acc;
}

template<uint R>
inline float dot_row16(threadgroup const float *A,
                       float4 q0,
                       float4 q1,
                       float4 q2,
                       float4 q3) {
    const uint base = R * 16u;
    float acc = 0.0f;
    acc = mad4_order(load4_tg(A + base + 0u),  q0, acc);
    acc = mad4_order(load4_tg(A + base + 4u),  q1, acc);
    acc = mad4_order(load4_tg(A + base + 8u),  q2, acc);
    acc = mad4_order(load4_tg(A + base + 12u), q3, acc);
    return acc;
}

template<bool COMPUTE_U>
inline float matvec_kick_d16_row(threadgroup const float *A,
                                 thread const float4 *q,
                                 thread float4 *p,
                                 float scale) {
    float4 q0 = q[0];
    float4 q1 = q[1];
    float4 q2 = q[2];
    float4 q3 = q[3];

    float U = 0.0f;

    {
        float4 f = float4(dot_row16<0u>(A, q0, q1, q2, q3),
                          dot_row16<1u>(A, q0, q1, q2, q3),
                          dot_row16<2u>(A, q0, q1, q2, q3),
                          dot_row16<3u>(A, q0, q1, q2, q3));
        if (COMPUTE_U) {
            U += 0.5f * q0.x * f.x;
            U += 0.5f * q0.y * f.y;
            U += 0.5f * q0.z * f.z;
            U += 0.5f * q0.w * f.w;
        }
        p[0] -= scale * f;
    }

    {
        float4 f = float4(dot_row16<4u>(A, q0, q1, q2, q3),
                          dot_row16<5u>(A, q0, q1, q2, q3),
                          dot_row16<6u>(A, q0, q1, q2, q3),
                          dot_row16<7u>(A, q0, q1, q2, q3));
        if (COMPUTE_U) {
            U += 0.5f * q1.x * f.x;
            U += 0.5f * q1.y * f.y;
            U += 0.5f * q1.z * f.z;
            U += 0.5f * q1.w * f.w;
        }
        p[1] -= scale * f;
    }

    {
        float4 f = float4(dot_row16<8u>(A, q0, q1, q2, q3),
                          dot_row16<9u>(A, q0, q1, q2, q3),
                          dot_row16<10u>(A, q0, q1, q2, q3),
                          dot_row16<11u>(A, q0, q1, q2, q3));
        if (COMPUTE_U) {
            U += 0.5f * q2.x * f.x;
            U += 0.5f * q2.y * f.y;
            U += 0.5f * q2.z * f.z;
            U += 0.5f * q2.w * f.w;
        }
        p[2] -= scale * f;
    }

    {
        float4 f = float4(dot_row16<12u>(A, q0, q1, q2, q3),
                          dot_row16<13u>(A, q0, q1, q2, q3),
                          dot_row16<14u>(A, q0, q1, q2, q3),
                          dot_row16<15u>(A, q0, q1, q2, q3));
        if (COMPUTE_U) {
            U += 0.5f * q3.x * f.x;
            U += 0.5f * q3.y * f.y;
            U += 0.5f * q3.z * f.z;
            U += 0.5f * q3.w * f.w;
        }
        p[3] -= scale * f;
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
    device const float4 *qin4 = reinterpret_cast<device const float4 *>(qin);
    device float4 *qout4 = reinterpret_cast<device float4 *>(qout);

    if (accept) {
#pragma unroll
        for (uint v = 0u; v < (D / 4u); ++v) {
            qout4[v] = q[v];
        }
    } else {
#pragma unroll
        for (uint v = 0u; v < (D / 4u); ++v) {
            qout4[v] = qin4[v];
        }
    }
}

template<uint D>
inline void run_hmc_fixed_chunk8(device const float *q_in,
                                 device float *q_out,
                                 device uint *accept_cnt,
                                 threadgroup const float *AT,
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

    device const float *qin = q_in + base;

    float K_old;
    init_fixed_state<D>(qin, q, p, K_old, base_counter, chain_seed);

    float half_eps = 0.5f * eps;

    float U_old = matvec_kick_chunk8<D, true>(AT, q, p, half_eps);

    float U_new = U_old;
    if (L != 0u) {
        uint full_steps = L - 1u;

        for (uint l = 0u; l < full_steps; ++l) {
            drift_fixed<D>(q, p, eps);
            (void)matvec_kick_chunk8<D, false>(AT, q, p, eps);
        }

        drift_fixed<D>(q, p, eps);
        U_new = matvec_kick_chunk8<D, true>(AT, q, p, half_eps);
    }

    float K_new = kinetic_fixed<D>(p);
    float dH = (U_new + K_new) - (U_old + K_old);

    bool accept = accept_hmc_chain(dH, chain_seed, base_counter + D);

    device float *qout = q_out + base;
    store_accept_fixed<D>(qin, qout, q, accept);

    if (accept) {
        accept_cnt[chain_idx] = accept_cnt[chain_idx] + 1u;
    }
}

inline void run_hmc_d16_row(device const float *q_in,
                            device float *q_out,
                            device uint *accept_cnt,
                            threadgroup const float *A,
                            uint L,
                            float eps,
                            uint hmc_step_idx,
                            uint seed,
                            uint chain_idx) {
    thread float4 q[4];
    thread float4 p[4];

    uint chain_seed = seed + chain_idx * 0x9E3779B9u;
    uint base_counter = hmc_step_idx * 17u;
    uint base = chain_idx * 16u;

    device const float *qin = q_in + base;

    float K_old;
    init_fixed_state<16u>(qin, q, p, K_old, base_counter, chain_seed);

    float half_eps = 0.5f * eps;

    float U_old = matvec_kick_d16_row<true>(A, q, p, half_eps);

    float U_new = U_old;
    if (L != 0u) {
        uint full_steps = L - 1u;

        for (uint l = 0u; l < full_steps; ++l) {
            drift_fixed<16u>(q, p, eps);
            (void)matvec_kick_d16_row<false>(A, q, p, eps);
        }

        drift_fixed<16u>(q, p, eps);
        U_new = matvec_kick_d16_row<true>(A, q, p, half_eps);
    }

    float K_new = kinetic_fixed<16u>(p);
    float dH = (U_new + K_new) - (U_old + K_old);

    bool accept = accept_hmc_chain(dH, chain_seed, base_counter + 16u);

    device float *qout = q_out + base;
    store_accept_fixed<16u>(qin, qout, q, accept);

    if (accept) {
        accept_cnt[chain_idx] = accept_cnt[chain_idx] + 1u;
    }
}

inline float matvec_kick_dynamic(threadgroup const float *AT,
                                 thread const float *q,
                                 thread float *p,
                                 uint d,
                                 float scale,
                                 bool computeU) {
    float U = 0.0f;

    for (uint i = 0u; i < d; ++i) {
        float acc = 0.0f;

        for (uint j = 0u; j < d; ++j) {
            acc += AT[j * d + i] * q[j];
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
                            threadgroup const float *AT,
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

    float U_old = matvec_kick_dynamic(AT, q, p, d, half_eps, true);

    float U_new = U_old;
    if (L != 0u) {
        uint full_steps = L - 1u;

        for (uint l = 0u; l < full_steps; ++l) {
            drift_dynamic(q, p, d, eps);
            (void)matvec_kick_dynamic(AT, q, p, d, eps, false);
        }

        drift_dynamic(q, p, d, eps);
        U_new = matvec_kick_dynamic(AT, q, p, d, half_eps, true);
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
    threadgroup float AT[1024];

    load_A_mixed_tg(A, AT, d, tid, tpg);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (chain_idx >= K) {
        return;
    }

    if (d == 8u) {
        run_hmc_fixed_chunk8<8u>(q_in, q_out, accept_cnt, AT, L, eps, hmc_step_idx, seed, chain_idx);
    } else if (d == 16u) {
        run_hmc_d16_row(q_in, q_out, accept_cnt, AT, L, eps, hmc_step_idx, seed, chain_idx);
    } else if (d == 24u) {
        run_hmc_fixed_chunk8<24u>(q_in, q_out, accept_cnt, AT, L, eps, hmc_step_idx, seed, chain_idx);
    } else if (d == 32u) {
        run_hmc_fixed_chunk8<32u>(q_in, q_out, accept_cnt, AT, L, eps, hmc_step_idx, seed, chain_idx);
    } else {
        run_hmc_dynamic(q_in, q_out, accept_cnt, AT, d, L, eps, hmc_step_idx, seed, chain_idx);
    }
}