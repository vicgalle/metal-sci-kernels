**Optimization:** Vectorize the inner mat-vec using `float4` loads from threadgroup memory. Since `d` is even and ≤ 32, we pad each row of the A tile to a multiple of 4 (stride = `(d+3)&~3`, max 32) and zero-fill the tail. Each thread's `q` array is also kept as `float4` chunks (mapped onto a flat `float[32]` so we can `*(threadgroup float4*)` load A and `*(thread float4*)` for q). This quarters the loop trip count, increases ILP via 4-wide FMAs, and improves threadgroup-memory bandwidth utilization. The mat-vec dominates runtime (`L*d²` FMAs per chain), so a 4× reduction in dependent loop iterations should translate directly to throughput. Box–Muller, accept/reject, and RNG are unchanged for bit-exact correctness.

```metal
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
    // Padded row stride (multiple of 4) so we can load float4 chunks safely.
    // d <= 32, so dpad <= 32.
    threadgroup float Atile[D_MAX * D_MAX];

    uint dpad = (d + 3u) & ~3u;

    // Cooperative load: zero-fill the tile, then place A[i,j] at Atile[i*dpad+j].
    uint tile_sz = D_MAX * D_MAX;
    for (uint k = tid_in_tg; k < tile_sz; k += tg_size) {
        Atile[k] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint dd = d * d;
    for (uint k = tid_in_tg; k < dd; k += tg_size) {
        uint i = k / d;
        uint j = k - i * d;
        Atile[i * dpad + j] = A[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (chain_idx >= K) return;

    // Use float arrays sized to a multiple of 4 so float4 loads are valid.
    thread float qbuf[D_MAX];
    thread float pbuf[D_MAX];
    thread float qold[D_MAX];
    thread float fbuf[D_MAX];

    // Zero-init tail (indices >= d) so float4 ops see 0 in padded slots.
    for (uint i = 0u; i < D_MAX; ++i) { qbuf[i] = 0.0f; }

    uint base_counter = hmc_step_idx * (d + 1u);
    float eps_half = 0.5f * eps;
    uint dpad4 = dpad >> 2;

    // 1. Momentum p ~ N(0, I), Box-Muller (counters i, i+1 for pair starting at i).
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
        pbuf[i] = r * c;
        if (i + 1u < d) {
            pbuf[i + 1u] = r * s;
        }
    }
    // Zero the padded tail of pbuf so dot products are unaffected.
    for (uint i = d; i < dpad; ++i) pbuf[i] = 0.0f;

    // 2. Load q, save q_old.
    for (uint i = 0u; i < d; ++i) {
        float qi = q_in[chain_idx * d + i];
        qbuf[i] = qi;
        qold[i] = qi;
    }
    // qbuf tail already 0 from init.

    // Helper: compute force = A * q via threadgroup-mem float4 loads.
    // Inlined below.

    // Initial force and U_old.
    float U_old = 0.0f;
    for (uint i = 0u; i < d; ++i) {
        threadgroup const float4 *Arow = (threadgroup const float4 *)(Atile + i * dpad);
        float4 acc4 = float4(0.0f);
        thread const float4 *qv = (thread const float4 *)qbuf;
        for (uint j4 = 0u; j4 < dpad4; ++j4) {
            acc4 = fma(Arow[j4], qv[j4], acc4);
        }
        float acc = acc4.x + acc4.y + acc4.z + acc4.w;
        fbuf[i] = acc;
        U_old = fma(0.5f * qbuf[i], acc, U_old);
    }
    // Pad fbuf tail.
    for (uint i = d; i < dpad; ++i) fbuf[i] = 0.0f;

    float K_old = 0.0f;
    for (uint i = 0u; i < d; ++i) K_old = fma(0.5f * pbuf[i], pbuf[i], K_old);

    // 3. Leapfrog.
    // Initial half-kick: p -= (eps/2) * force.
    {
        thread float4 *pv = (thread float4 *)pbuf;
        thread const float4 *fv = (thread const float4 *)fbuf;
        for (uint j4 = 0u; j4 < dpad4; ++j4) {
            pv[j4] = fma(float4(-eps_half), fv[j4], pv[j4]);
        }
    }

    for (uint l = 0u; l < L; ++l) {
        // Drift: q += eps * p
        {
            thread float4 *qv = (thread float4 *)qbuf;
            thread const float4 *pv = (thread const float4 *)pbuf;
            for (uint j4 = 0u; j4 < dpad4; ++j4) {
                qv[j4] = fma(float4(eps), pv[j4], qv[j4]);
            }
        }
        // Recompute force = A q.
        for (uint i = 0u; i < d; ++i) {
            threadgroup const float4 *Arow = (threadgroup const float4 *)(Atile + i * dpad);
            float4 acc4 = float4(0.0f);
            thread const float4 *qv = (thread const float4 *)qbuf;
            for (uint j4 = 0u; j4 < dpad4; ++j4) {
                acc4 = fma(Arow[j4], qv[j4], acc4);
            }
            fbuf[i] = acc4.x + acc4.y + acc4.z + acc4.w;
        }
        float scale = (l + 1u == L) ? eps_half : eps;
        {
            thread float4 *pv = (thread float4 *)pbuf;
            thread const float4 *fv = (thread const float4 *)fbuf;
            float4 sc4 = float4(-scale);
            for (uint j4 = 0u; j4 < dpad4; ++j4) {
                pv[j4] = fma(sc4, fv[j4], pv[j4]);
            }
        }
    }

    // 4. New Hamiltonian (reuse fbuf).
    float U_new = 0.0f;
    for (uint i = 0u; i < d; ++i) U_new = fma(0.5f * qbuf[i], fbuf[i], U_new);
    float K_new = 0.0f;
    for (uint i = 0u; i < d; ++i) K_new = fma(0.5f * pbuf[i], pbuf[i], K_new);

    // 5. Accept/reject.
    float dH = (U_new + K_new) - (U_old + K_old);
    uint b_acc = rand_u32(seed, chain_idx, base_counter + d);
    float u_acc = u01_from_bits(b_acc);
    float log_u = log(max(u_acc, 1.0e-30f));
    bool accept = isfinite(dH) && (log_u < -dH);

    for (uint i = 0u; i < d; ++i) {
        q_out[chain_idx * d + i] = accept ? qbuf[i] : qold[i];
    }
    if (accept) {
        accept_cnt[chain_idx] = accept_cnt[chain_idx] + 1u;
    }
}
```