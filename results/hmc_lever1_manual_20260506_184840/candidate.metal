// HMC candidate: register-blocked over multiple chains per thread (the
// "n-body lever").
//
// Iter 6's win was template specialization on D so the matvec fully
// unrolls. This candidate adds: each thread now processes NC chains.
// The same A[i,j] entry, loaded once from threadgroup memory, drives
// NC parallel FMA chains -- one per chain. Two effects:
//   1. NC-fold ILP across the j loop (NC independent acc accumulators).
//   2. NC-fold reuse of every threadgroup-A read.
//
// NC is chosen per-d to balance ILP gain against per-thread register
// pressure:
//   d= 8: NC=4   (4*32 = 128 floats per thread)
//   d=16: NC=2   (2*64 = 128 floats per thread)
//   d=32: NC=1   (state already 128 floats; doubling spills hard)
//
// Host dispatches K threads; each thread handles chains
// [chain_idx * NC, chain_idx * NC + NC). The (NC-1)/NC fraction of
// threads with chain_base >= K early-exit immediately. Indexing is
// strided so that the first K/NC simdgroups are fully active and the
// rest exit, keeping good locality on q_in / q_out.

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

template <uint D, uint NC>
inline void hmc_run_blocked(uint chain_base,
                             uint K,
                             device const float *q_in,
                             device       float *q_out,
                             device       uint  *accept_cnt,
                             threadgroup const float *Atile,
                             uint L, float eps,
                             uint hmc_step_idx, uint seed)
{
    float q[NC][D];
    float p[NC][D];
    float f[NC][D];
    float qold[NC][D];
    bool active[NC];

    uint base_counter = hmc_step_idx * (D + 1u);
    float eps_half = 0.5f * eps;

    #pragma unroll
    for (uint c = 0u; c < NC; ++c) {
        active[c] = (chain_base + c) < K;
    }

    // 1) Box-Muller for each chain. Inactive chains use cidx=chain_base
    // (a valid index < K, since we early-exited if chain_base >= K) so
    // their RNG calls produce sane values; their results are discarded
    // at write time.
    #pragma unroll
    for (uint c = 0u; c < NC; ++c) {
        uint cidx = active[c] ? (chain_base + c) : chain_base;
        #pragma unroll
        for (uint i = 0u; i < D; i += 2u) {
            uint b1 = rand_u32(seed, cidx, base_counter + i);
            uint b2 = rand_u32(seed, cidx, base_counter + i + 1u);
            float u1 = u01_from_bits(b1);
            float u2 = u01_from_bits(b2);
            u1 = max(u1, 1.0e-7f);
            float r = sqrt(-2.0f * log(u1));
            float angle = 6.2831853071795864f * u2;
            float cv;
            float sv = sincos(angle, cv);
            p[c][i] = r * cv;
            p[c][i + 1u] = r * sv;
        }
    }

    // 2) Load q, save qold. Inactive chains read chain 0's row -- valid
    // memory, garbage in the rest of this kernel for those slots.
    #pragma unroll
    for (uint c = 0u; c < NC; ++c) {
        uint cidx = active[c] ? (chain_base + c) : 0u;
        device const float *qin = q_in + cidx * D;
        #pragma unroll
        for (uint i = 0u; i < D; ++i) {
            float qi = qin[i];
            q[c][i] = qi;
            qold[c][i] = qi;
        }
    }

    // matvec f = A q -- this is where the lever pays off: the load of
    // Atile[i*D + j] is amortised across NC parallel FMA chains.
    #pragma unroll
    for (uint i = 0u; i < D; ++i) {
        threadgroup const float *Arow = Atile + i * D;
        float acc[NC];
        #pragma unroll
        for (uint c = 0u; c < NC; ++c) acc[c] = 0.0f;
        #pragma unroll
        for (uint j = 0u; j < D; ++j) {
            float Aij = Arow[j];
            #pragma unroll
            for (uint c = 0u; c < NC; ++c) {
                acc[c] = fma(Aij, q[c][j], acc[c]);
            }
        }
        #pragma unroll
        for (uint c = 0u; c < NC; ++c) f[c][i] = acc[c];
    }

    float U_old[NC];
    float K_old[NC];
    #pragma unroll
    for (uint c = 0u; c < NC; ++c) { U_old[c] = 0.0f; K_old[c] = 0.0f; }
    #pragma unroll
    for (uint i = 0u; i < D; ++i) {
        #pragma unroll
        for (uint c = 0u; c < NC; ++c) {
            U_old[c] = fma(q[c][i], f[c][i], U_old[c]);
            K_old[c] = fma(p[c][i], p[c][i], K_old[c]);
        }
    }
    #pragma unroll
    for (uint c = 0u; c < NC; ++c) { U_old[c] *= 0.5f; K_old[c] *= 0.5f; }

    // 3) Leapfrog. Initial half-kick.
    #pragma unroll
    for (uint i = 0u; i < D; ++i) {
        #pragma unroll
        for (uint c = 0u; c < NC; ++c) {
            p[c][i] = fma(-eps_half, f[c][i], p[c][i]);
        }
    }

    for (uint l = 0u; l < L; ++l) {
        // drift
        #pragma unroll
        for (uint i = 0u; i < D; ++i) {
            #pragma unroll
            for (uint c = 0u; c < NC; ++c) {
                q[c][i] = fma(eps, p[c][i], q[c][i]);
            }
        }
        // matvec (NC-blocked)
        #pragma unroll
        for (uint i = 0u; i < D; ++i) {
            threadgroup const float *Arow = Atile + i * D;
            float acc[NC];
            #pragma unroll
            for (uint c = 0u; c < NC; ++c) acc[c] = 0.0f;
            #pragma unroll
            for (uint j = 0u; j < D; ++j) {
                float Aij = Arow[j];
                #pragma unroll
                for (uint c = 0u; c < NC; ++c) {
                    acc[c] = fma(Aij, q[c][j], acc[c]);
                }
            }
            #pragma unroll
            for (uint c = 0u; c < NC; ++c) f[c][i] = acc[c];
        }
        // kick (full step except final half)
        float scale = (l + 1u == L) ? -eps_half : -eps;
        #pragma unroll
        for (uint i = 0u; i < D; ++i) {
            #pragma unroll
            for (uint c = 0u; c < NC; ++c) {
                p[c][i] = fma(scale, f[c][i], p[c][i]);
            }
        }
    }

    float U_new[NC];
    float K_new[NC];
    #pragma unroll
    for (uint c = 0u; c < NC; ++c) { U_new[c] = 0.0f; K_new[c] = 0.0f; }
    #pragma unroll
    for (uint i = 0u; i < D; ++i) {
        #pragma unroll
        for (uint c = 0u; c < NC; ++c) {
            U_new[c] = fma(q[c][i], f[c][i], U_new[c]);
            K_new[c] = fma(p[c][i], p[c][i], K_new[c]);
        }
    }
    #pragma unroll
    for (uint c = 0u; c < NC; ++c) { U_new[c] *= 0.5f; K_new[c] *= 0.5f; }

    // 5) Accept/reject and write. Skip inactive chains entirely.
    #pragma unroll
    for (uint c = 0u; c < NC; ++c) {
        if (!active[c]) continue;
        uint cidx = chain_base + c;
        float dH = (U_new[c] + K_new[c]) - (U_old[c] + K_old[c]);
        uint b_acc = rand_u32(seed, cidx, base_counter + D);
        float u_acc = u01_from_bits(b_acc);
        float log_u = log(max(u_acc, 1.0e-30f));
        bool accept = isfinite(dH) && (log_u < -dH);

        device float *qout = q_out + cidx * D;
        #pragma unroll
        for (uint i = 0u; i < D; ++i) {
            qout[i] = accept ? q[c][i] : qold[c][i];
        }
        if (accept) {
            accept_cnt[cidx] = accept_cnt[cidx] + 1u;
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
                     uint tid_in_tg [[thread_position_in_threadgroup]],
                     uint tg_size   [[threads_per_threadgroup]])
{
    threadgroup float Atile[32u * 32u];
    uint dd = d * d;
    for (uint k = tid_in_tg; k < dd; k += tg_size) {
        Atile[k] = A[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (d == 8u) {
        constexpr uint NC = 2u;
        uint chain_base = chain_idx * NC;
        if (chain_base >= K) return;
        hmc_run_blocked<8u, NC>(chain_base, K, q_in, q_out, accept_cnt,
                                Atile, L, eps, hmc_step_idx, seed);
    } else if (d == 16u) {
        constexpr uint NC = 2u;
        uint chain_base = chain_idx * NC;
        if (chain_base >= K) return;
        hmc_run_blocked<16u, NC>(chain_base, K, q_in, q_out, accept_cnt,
                                 Atile, L, eps, hmc_step_idx, seed);
    } else {
        // d == 32: NC=1 (per-thread state already at the limit).
        if (chain_idx >= K) return;
        hmc_run_blocked<32u, 1u>(chain_idx, K, q_in, q_out, accept_cnt,
                                 Atile, L, eps, hmc_step_idx, seed);
    }
}
