## Task: hmc

Hamiltonian Monte Carlo on a multivariate Gaussian target with mean 0 and precision matrix A = Sigma^{-1} (provided as a (d, d) row-major float32 buffer). One thread per chain; many chains run in parallel.

Per HMC step (one dispatch):
  1) p ~ N(0, I): for each pair (i, i+1) in 0..d-1 step 2,
     draw two uniforms u1, u2 in [0, 1) via the prescribed RNG
     (counters base_counter + i and base_counter + i + 1, where
     base_counter = hmc_step_idx * (d + 1)) and apply Box-Muller:
        u1 = max(u1, 1e-7);  r = sqrt(-2 * log(u1));
        angle = 2 pi * u2;
        p[i]   = r * cos(angle);
        p[i+1] = r * sin(angle);   // skip if i+1 >= d (d is even).
  2) Save q_old = q. Compute force = A q;
     U_old = 0.5 * dot(q, force); K_old = 0.5 * dot(p, p).
  3) Leapfrog with eps:
        p   -= (eps/2) * force            // initial half-kick
        for l = 0..L-1:
            q   += eps * p                // drift
            force = A q                   // recompute force at new q
            scale = (l + 1 == L) ? (eps/2) : eps
            p   -= scale * force          // kick
  4) U_new = 0.5 * dot(q, force) [reusing the final force];
     K_new = 0.5 * dot(p, p);
     dH = (U_new + K_new) - (U_old + K_old).
  5) Draw uniform u_acc with counter base_counter + d.
     accept = isfinite(dH) AND log(max(u_acc, 1e-30)) < -dH.
     Write q if accept else q_old to q_out[chain_idx * d + i];
     if accept, accept_cnt[chain_idx] += 1.

RNG (must be reproduced bit-exactly):
  inline uint mix32(uint x) {
      x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
      x = (x ^ (x >> 13)) * 0xC2B2AE35u;
      return x ^ (x >> 16);
  }
  uint x = seed + chain_idx * 0x9E3779B9u;
  x = mix32(x ^ counter);
  x = mix32(x);
  float u = float(x >> 8) * (1.0f / 16777216.0f);

The host ping-pongs (q_in, q_out) buffers across HMC steps; all dispatches share one command buffer for end-to-end timing.

## Required kernel signature(s)

```
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
                     uint chain_idx [[thread_position_in_grid]]);

Threads are dispatched 1-D, one per chain; guard with `if (chain_idx >= K) return;`. The host ping-pongs (q_in, q_out) between two K * d float buffers and increments hmc_step_idx by 1 per dispatch. accept_cnt is initialised to zero and accumulates accepted proposals over the run.

All chosen sizes satisfy d <= 32 and d is even; thread-private arrays of size 32 are sufficient. Threadgroup-cooperative schemes (multiple threads per chain sharing the mat-vec) and simdgroup reductions are valid optimisations as long as the external buffer layout above is preserved.
```

## Your previous attempt

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

template<bool COMPUTE_U>
inline float matvec_kick_d16(threadgroup const float *AT,
                             thread const float4 *q,
                             thread float4 *p,
                             float scale) {
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float4 acc3 = float4(0.0f);

#pragma unroll
    for (uint v = 0u; v < 4u; ++v) {
        float4 qv = q[v];
        uint j = v << 2;

        threadgroup const float *c0 = AT + (j + 0u) * 16u;
        threadgroup const float *c1 = AT + (j + 1u) * 16u;
        threadgroup const float *c2 = AT + (j + 2u) * 16u;
        threadgroup const float *c3 = AT + (j + 3u) * 16u;

        acc0 += load4_tg(c0)       * qv.x;
        acc1 += load4_tg(c0 + 4u)  * qv.x;
        acc2 += load4_tg(c0 + 8u)  * qv.x;
        acc3 += load4_tg(c0 + 12u) * qv.x;

        acc0 += load4_tg(c1)       * qv.y;
        acc1 += load4_tg(c1 + 4u)  * qv.y;
        acc2 += load4_tg(c1 + 8u)  * qv.y;
        acc3 += load4_tg(c1 + 12u) * qv.y;

        acc0 += load4_tg(c2)       * qv.z;
        acc1 += load4_tg(c2 + 4u)  * qv.z;
        acc2 += load4_tg(c2 + 8u)  * qv.z;
        acc3 += load4_tg(c2 + 12u) * qv.z;

        acc0 += load4_tg(c3)       * qv.w;
        acc1 += load4_tg(c3 + 4u)  * qv.w;
        acc2 += load4_tg(c3 + 8u)  * qv.w;
        acc3 += load4_tg(c3 + 12u) * qv.w;
    }

    float U = 0.0f;

    if (COMPUTE_U) {
        float4 q0 = q[0];
        float4 q1 = q[1];
        float4 q2 = q[2];
        float4 q3 = q[3];

        U += 0.5f * q0.x * acc0.x;
        U += 0.5f * q0.y * acc0.y;
        U += 0.5f * q0.z * acc0.z;
        U += 0.5f * q0.w * acc0.w;

        U += 0.5f * q1.x * acc1.x;
        U += 0.5f * q1.y * acc1.y;
        U += 0.5f * q1.z * acc1.z;
        U += 0.5f * q1.w * acc1.w;

        U += 0.5f * q2.x * acc2.x;
        U += 0.5f * q2.y * acc2.y;
        U += 0.5f * q2.z * acc2.z;
        U += 0.5f * q2.w * acc2.w;

        U += 0.5f * q3.x * acc3.x;
        U += 0.5f * q3.y * acc3.y;
        U += 0.5f * q3.z * acc3.z;
        U += 0.5f * q3.w * acc3.w;
    }

    p[0] -= scale * acc0;
    p[1] -= scale * acc1;
    p[2] -= scale * acc2;
    p[3] -= scale * acc3;

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

inline void run_hmc_d16(device const float *q_in,
                        device float *q_out,
                        device uint *accept_cnt,
                        threadgroup const float *AT,
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

    float U_old = matvec_kick_d16<true>(AT, q, p, half_eps);

    float U_new = U_old;
    if (L != 0u) {
        uint full_steps = L - 1u;

        for (uint l = 0u; l < full_steps; ++l) {
            drift_fixed<16u>(q, p, eps);
            (void)matvec_kick_d16<false>(AT, q, p, eps);
        }

        drift_fixed<16u>(q, p, eps);
        U_new = matvec_kick_d16<true>(AT, q, p, half_eps);
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

    load_A_transpose_tg(A, AT, d, tid, tpg);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (chain_idx >= K) {
        return;
    }

    if (d == 8u) {
        run_hmc_fixed_chunk8<8u>(q_in, q_out, accept_cnt, AT, L, eps, hmc_step_idx, seed, chain_idx);
    } else if (d == 16u) {
        run_hmc_d16(q_in, q_out, accept_cnt, AT, L, eps, hmc_step_idx, seed, chain_idx);
    } else if (d == 24u) {
        run_hmc_fixed_chunk8<24u>(q_in, q_out, accept_cnt, AT, L, eps, hmc_step_idx, seed, chain_idx);
    } else if (d == 32u) {
        run_hmc_fixed_chunk8<32u>(q_in, q_out, accept_cnt, AT, L, eps, hmc_step_idx, seed, chain_idx);
    } else {
        run_hmc_dynamic(q_in, q_out, accept_cnt, AT, d, L, eps, hmc_step_idx, seed, chain_idx);
    }
}
```

Result of previous attempt:
         d8_K16384: correct, 4.31 ms, 1004.5 GFLOPS (mat-vec FMAs only) (22.3% of 4500 GFLOPS)
         d16_K4096: correct, 40.56 ms, 110.2 GFLOPS (mat-vec FMAs only) (2.4% of 4500 GFLOPS)
         d32_K1024: correct, 21.69 ms, 209.4 GFLOPS (mat-vec FMAs only) (4.7% of 4500 GFLOPS)
  score (gmean of fraction): 0.0634

## History

- iter  0: compile=OK | correct=True | score=0.008809128380504386
- iter  1: compile=FAIL | correct=False | score=N/A
- iter  2: compile=OK | correct=True | score=0.05854090261728891
- iter  3: compile=FAIL | correct=False | score=N/A
- iter  4: compile=OK | correct=True | score=0.027658441989953843
- iter  5: compile=FAIL | correct=False | score=N/A
- iter  6: compile=OK | correct=True | score=0.06336697071045806

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
