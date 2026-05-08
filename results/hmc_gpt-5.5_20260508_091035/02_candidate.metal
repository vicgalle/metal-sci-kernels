#include <metal_stdlib>
using namespace metal;

#define D_MAX 32u

inline uint mix32(uint x) {
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    return x ^ (x >> 16);
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

inline float4 load4_tg(threadgroup const float *p) {
    return float4(p[0], p[1], p[2], p[3]);
}

inline void load_A_transpose_tg(device const float *A,
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
            uint row = idx >> 4;
            uint col = idx & 15u;
            AT[(col << 4) + row] = A[idx];
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

inline bool accept_hmc(float dH, uint seed, uint chain_idx, uint acc_counter) {
    if (!isfinite(dH)) {
        return false;
    }
    if (dH <= 0.0f) {
        return true;
    }

    uint b_acc = rand_u32(seed, chain_idx, acc_counter);
    float u_acc = u01_from_bits(b_acc);
    float log_u = log(max(u_acc, 1.0e-30f));
    return log_u < -dH;
}

template<uint D, bool COMPUTE_U>
inline float matvec_kick_fixed(threadgroup const float *AT,
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

            acc0 += load4_tg(c0) * qv.x;
            acc1 += load4_tg(c0 + 4u) * qv.x;

            acc0 += load4_tg(c1) * qv.y;
            acc1 += load4_tg(c1 + 4u) * qv.y;

            acc0 += load4_tg(c2) * qv.z;
            acc1 += load4_tg(c2 + 4u) * qv.z;

            acc0 += load4_tg(c3) * qv.w;
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
inline void run_hmc_fixed(device const float *q_in,
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

    uint base_counter = hmc_step_idx * (D + 1u);
    uint base = chain_idx * D;

    float K_old = 0.0f;

#pragma unroll
    for (uint i = 0u; i < D; i += 2u) {
        uint b1 = rand_u32(seed, chain_idx, base_counter + i);
        uint b2 = rand_u32(seed, chain_idx, base_counter + i + 1u);

        float u1 = max(u01_from_bits(b1), 1.0e-7f);
        float u2 = u01_from_bits(b2);

        float r = sqrt(-2.0f * log(u1));
        float angle = 6.2831853071795864f * u2;

        float c, s;
        s = sincos(angle, c);

        float p0 = r * c;
        float p1 = r * s;

        uint v = i >> 2;
        if ((i & 2u) == 0u) {
            p[v].x = p0;
            p[v].y = p1;
        } else {
            p[v].z = p0;
            p[v].w = p1;
        }

        K_old += 0.5f * p0 * p0;
        K_old += 0.5f * p1 * p1;
    }

    device const float *qin = q_in + base;

#pragma unroll
    for (uint v = 0u; v < (D / 4u); ++v) {
        uint o = v << 2;
        q[v] = float4(qin[o + 0u], qin[o + 1u], qin[o + 2u], qin[o + 3u]);
    }

    float half_eps = 0.5f * eps;

    float U_old = matvec_kick_fixed<D, true>(AT, q, p, half_eps);

    float U_new = U_old;
    if (L != 0u) {
        uint full_steps = L - 1u;

        for (uint l = 0u; l < full_steps; ++l) {
            drift_fixed<D>(q, p, eps);
            (void)matvec_kick_fixed<D, false>(AT, q, p, eps);
        }

        drift_fixed<D>(q, p, eps);
        U_new = matvec_kick_fixed<D, true>(AT, q, p, half_eps);
    }

    float K_new = kinetic_fixed<D>(p);
    float dH = (U_new + K_new) - (U_old + K_old);

    bool accept = accept_hmc(dH, seed, chain_idx, base_counter + D);

    device float *qout = q_out + base;
    store_accept_fixed<D>(qin, qout, q, accept);

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

    uint base_counter = hmc_step_idx * (d + 1u);
    uint base = chain_idx * d;

    float K_old = 0.0f;

    for (uint i = 0u; i < d; i += 2u) {
        uint b1 = rand_u32(seed, chain_idx, base_counter + i);
        uint b2 = rand_u32(seed, chain_idx, base_counter + i + 1u);

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

    bool accept = accept_hmc(dH, seed, chain_idx, base_counter + d);

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

    load_A_transpose_tg(A, AT, d, tid, tpg);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (chain_idx >= K) {
        return;
    }

    if (d == 8u) {
        run_hmc_fixed<8u>(q_in, q_out, accept_cnt, AT, L, eps, hmc_step_idx, seed, chain_idx);
    } else if (d == 16u) {
        run_hmc_fixed<16u>(q_in, q_out, accept_cnt, AT, L, eps, hmc_step_idx, seed, chain_idx);
    } else if (d == 24u) {
        run_hmc_fixed<24u>(q_in, q_out, accept_cnt, AT, L, eps, hmc_step_idx, seed, chain_idx);
    } else if (d == 32u) {
        run_hmc_fixed<32u>(q_in, q_out, accept_cnt, AT, L, eps, hmc_step_idx, seed, chain_idx);
    } else {
        run_hmc_dynamic(q_in, q_out, accept_cnt, AT, d, L, eps, hmc_step_idx, seed, chain_idx);
    }
}