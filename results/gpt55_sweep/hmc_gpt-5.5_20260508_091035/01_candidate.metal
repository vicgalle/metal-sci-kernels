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

template<uint D, bool COMPUTE_U>
inline float matvec_kick_fixed(threadgroup const float *AT,
                               thread const float *q,
                               thread float *p,
                               float scale) {
    float U = 0.0f;

#pragma unroll
    for (uint r = 0u; r < D; r += 8u) {
        float4 acc0 = float4(0.0f);
        float4 acc1 = float4(0.0f);

#pragma unroll
        for (uint j = 0u; j < D; ++j) {
            float qj = q[j];
            threadgroup const float *col = AT + j * D + r;
            float4 a0 = load4_tg(col);
            float4 a1 = load4_tg(col + 4u);
            acc0 += a0 * qj;
            acc1 += a1 * qj;
        }

        if (COMPUTE_U) {
            U += 0.5f * q[r + 0u] * acc0.x;
            U += 0.5f * q[r + 1u] * acc0.y;
            U += 0.5f * q[r + 2u] * acc0.z;
            U += 0.5f * q[r + 3u] * acc0.w;
            U += 0.5f * q[r + 4u] * acc1.x;
            U += 0.5f * q[r + 5u] * acc1.y;
            U += 0.5f * q[r + 6u] * acc1.z;
            U += 0.5f * q[r + 7u] * acc1.w;
        }

        p[r + 0u] -= scale * acc0.x;
        p[r + 1u] -= scale * acc0.y;
        p[r + 2u] -= scale * acc0.z;
        p[r + 3u] -= scale * acc0.w;
        p[r + 4u] -= scale * acc1.x;
        p[r + 5u] -= scale * acc1.y;
        p[r + 6u] -= scale * acc1.z;
        p[r + 7u] -= scale * acc1.w;
    }

    return U;
}

template<uint D>
inline void drift_fixed(thread float *q, thread const float *p, float eps) {
#pragma unroll
    for (uint i = 0u; i < D; ++i) {
        q[i] += eps * p[i];
    }
}

template<uint D>
inline float kinetic_fixed(thread const float *p) {
    float K = 0.0f;
#pragma unroll
    for (uint i = 0u; i < D; ++i) {
        K += 0.5f * p[i] * p[i];
    }
    return K;
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
    thread float q[D_MAX];
    thread float p[D_MAX];

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
        p[i] = p0;
        p[i + 1u] = p1;
        K_old += 0.5f * p0 * p0;
        K_old += 0.5f * p1 * p1;
    }

    device const float *qin = q_in + base;
#pragma unroll
    for (uint i = 0u; i < D; ++i) {
        q[i] = qin[i];
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
    if (accept) {
#pragma unroll
        for (uint i = 0u; i < D; ++i) {
            qout[i] = q[i];
        }
        accept_cnt[chain_idx] = accept_cnt[chain_idx] + 1u;
    } else {
#pragma unroll
        for (uint i = 0u; i < D; ++i) {
            qout[i] = qin[i];
        }
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
    float K = 0.0f;
    for (uint i = 0u; i < d; ++i) {
        K += 0.5f * p[i] * p[i];
    }
    return K;
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
                     uint3 tpg3     [[threads_per_threadgroup]]) [[max_total_threads_per_threadgroup(32)]] {
    threadgroup float AT[1024];

    uint tpg = tpg3.x;
    load_A_transpose_tg(A, AT, d, tid, tpg);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (chain_idx >= K) {
        return;
    }

    if (d == 8u) {
        run_hmc_fixed<8u>(q_in, q_out, accept_cnt, AT, L, eps, hmc_step_idx, seed, chain_idx);
    } else if (d == 16u) {
        run_hmc_fixed<16u>(q_in, q_out, accept_cnt, AT, L, eps, hmc_step_idx, seed, chain_idx);
    } else if (d == 32u) {
        run_hmc_fixed<32u>(q_in, q_out, accept_cnt, AT, L, eps, hmc_step_idx, seed, chain_idx);
    } else {
        run_hmc_dynamic(q_in, q_out, accept_cnt, AT, d, L, eps, hmc_step_idx, seed, chain_idx);
    }
}