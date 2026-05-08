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
```

Result of previous attempt:
  COMPILE FAILED: Error Domain=MTLLibraryErrorDomain Code=3 "program_source:330:68: error: 'max_total_threads_per_threadgroup' attribute cannot be applied to types
                     uint3 tpg3     [[threads_per_threadgroup]]) [[max_total_threads_per_threadgroup(32)]] {
                                                                   ^
program_source:318:13: error: expecting input declarations with either all scalar types or all vector types with the same number of elements
kernel void hmc_step(device const float *q_in        [[buffer(0)]],
            ^
program_source:328:27: note: declaration with attribute 'thread_position_in_grid' of type 'uint' (aka 'unsigned int') here
                     uint chain_idx [[thread_position_in_grid]],
                     ~~~~~^~~~~~~~~
program_source:330:28: note: declaration with attribute 'threads_per_threadgroup' of type 'uint3' (vector of 3 'unsigned int' values) here
                     uint3 tpg3     [[threads_per_threadgroup]]) [[max_total_threads_per_threadgroup(32)]] {
                     ~~~~~~^~~~
" UserInfo={NSLocalizedDescription=program_source:330:68: error: 'max_total_threads_per_threadgroup' attribute cannot be applied to types
                     uint3 tpg3     [[threads_per_threadgroup]]) [[max_total_threads_per_threadgroup(32)]] {
                                                                   ^
program_source:318:13: error: expecting input declarations with either all scalar types or all vector types with the same number of elements
kernel void hmc_step(device const float *q_in        [[buffer(0)]],
            ^
program_source:328:27: note: declaration with attribute 'thread_position_in_grid' of type 'uint' (aka 'unsigned int') here
                     uint chain_idx [[thread_position_in_grid]],
                     ~~~~~^~~~~~~~~
program_source:330:28: note: declaration with attribute 'threads_per_threadgroup' of type 'uint3' (vector of 3 'unsigned int' values) here
                     uint3 tpg3     [[threads_per_threadgroup]]) [[max_total_threads_per_threadgroup(32)]] {
                     ~~~~~~^~~~
}

## Current best (incumbent)

```metal
// Naive seed kernel for Hamiltonian Monte Carlo on a Gaussian target.
// One thread per chain.
//
// Target: U(q) = (1/2) q^T A q  with A = Sigma^{-1} symmetric PD.
// Per HMC step (one dispatch):
//   1. Sample momentum p ~ N(0, I) via Box-Muller (d Gaussians per chain).
//   2. Save q_old. Compute U_old, K_old.
//   3. Leapfrog with step size eps for L steps:
//        p   <- p - (eps/2) A q                  // initial half-kick
//        for l = 1..L-1:
//            q <- q + eps p                       // drift
//            p <- p - eps A q                     // full kick
//        q   <- q + eps p                         // last drift
//        p   <- p - (eps/2) A q                   // final half-kick
//   4. Compute U_new, K_new. dH = (U_new + K_new) - (U_old + K_old).
//   5. Accept iff log(u_acc) < -dH (equivalently u_acc < exp(-dH));
//      otherwise rollback q <- q_old. Increment per-chain accept count
//      on accept.
//
// RNG: Murmur3-fmix32 hash of (seed, chain_idx, base_counter + sample_idx)
// where base_counter = hmc_step_idx * (d + 1). Per HMC step per chain we
// need d uniforms for the momentum (paired into d/2 Box-Muller calls)
// plus 1 uniform for accept.
//
// Per-thread storage uses thread-private arrays sized to a compile-time
// upper bound D_MAX. The host's sizes all satisfy d <= D_MAX = 32.
//
// Buffer layout (must be preserved by candidate kernels):
//   buffer 0: device const float *q_in        (K * d)
//   buffer 1: device       float *q_out       (K * d)
//   buffer 2: device       uint  *accept_cnt  (K, accumulates over HMC steps)
//   buffer 3: device const float *A           (d * d, row-major precision matrix)
//   buffer 4: const uint  &K
//   buffer 5: const uint  &d
//   buffer 6: const uint  &L
//   buffer 7: const float &eps
//   buffer 8: const uint  &hmc_step_idx
//   buffer 9: const uint  &seed

#include <metal_stdlib>
using namespace metal;

#define D_MAX 32u

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
    // 24-bit uniform in [0, 1); exact in fp32 on every platform.
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
                     uint chain_idx [[thread_position_in_grid]]) {
    if (chain_idx >= K) return;

    thread float q[D_MAX];
    thread float p[D_MAX];
    thread float q_old[D_MAX];
    thread float force[D_MAX];

    uint base_counter = hmc_step_idx * (d + 1u);

    // 1. Momentum p ~ N(0, I), Box-Muller over (u1, u2) pairs.
    for (uint i = 0u; i < d; i += 2u) {
        uint b1 = rand_u32(seed, chain_idx, base_counter + i);
        uint b2 = rand_u32(seed, chain_idx, base_counter + i + 1u);
        float u1 = u01_from_bits(b1);
        float u2 = u01_from_bits(b2);
        u1 = max(u1, 1.0e-7f);                           // guard log(0)
        float r = sqrt(-2.0f * log(u1));
        float angle = 6.2831853071795864f * u2;          // 2 pi
        float c, s;
        s = sincos(angle, c);
        p[i] = r * c;
        if (i + 1u < d) {
            p[i + 1u] = r * s;
        }
    }

    // 2. Load q from q_in, save q_old.
    for (uint i = 0u; i < d; ++i) {
        float qi = q_in[chain_idx * d + i];
        q[i] = qi;
        q_old[i] = qi;
    }

    // Helper: force[i] = sum_j A[i, j] * q[j]; folds U = (1/2) q . force.
    float U_old = 0.0f;
    for (uint i = 0u; i < d; ++i) {
        float acc = 0.0f;
        for (uint j = 0u; j < d; ++j) {
            acc += A[i * d + j] * q[j];
        }
        force[i] = acc;
        U_old += 0.5f * q[i] * acc;
    }
    float K_old = 0.0f;
    for (uint i = 0u; i < d; ++i) K_old += 0.5f * p[i] * p[i];

    // 3. Leapfrog. We already have force = A q at q_0, so use it for
    //    the initial half-kick without a redundant mat-vec.
    for (uint i = 0u; i < d; ++i) p[i] -= 0.5f * eps * force[i];

    for (uint l = 0u; l < L; ++l) {
        // drift
        for (uint i = 0u; i < d; ++i) q[i] += eps * p[i];
        // recompute force at new q
        for (uint i = 0u; i < d; ++i) {
            float acc = 0.0f;
            for (uint j = 0u; j < d; ++j) acc += A[i * d + j] * q[j];
            force[i] = acc;
        }
        // kick (full step except after the last drift, where it's a half)
        float scale = (l + 1u == L) ? (0.5f * eps) : eps;
        for (uint i = 0u; i < d; ++i) p[i] -= scale * force[i];
    }

    // 4. New Hamiltonian. We have force = A q at the final q.
    float U_new = 0.0f;
    for (uint i = 0u; i < d; ++i) U_new += 0.5f * q[i] * force[i];
    float K_new = 0.0f;
    for (uint i = 0u; i < d; ++i) K_new += 0.5f * p[i] * p[i];

    // 5. Accept iff log(u_acc) < -dH (handles dH < 0 and dH > 0
    // uniformly; avoids exp() overflow on rejected proposals).
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
```

Incumbent result:
         d8_K16384: correct, 43.21 ms, 100.1 GFLOPS (mat-vec FMAs only) (2.2% of 4500 GFLOPS)
         d16_K4096: correct, 102.90 ms, 43.4 GFLOPS (mat-vec FMAs only) (1.0% of 4500 GFLOPS)
         d32_K1024: correct, 317.03 ms, 14.3 GFLOPS (mat-vec FMAs only) (0.3% of 4500 GFLOPS)
  score (gmean of fraction): 0.0088

## History

- iter  0: compile=OK | correct=True | score=0.008809128380504386
- iter  1: compile=FAIL | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
