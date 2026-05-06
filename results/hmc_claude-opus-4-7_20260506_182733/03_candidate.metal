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

[[max_total_threads_per_threadgroup(256)]]
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

    // Thread-private state, padded to multiple of 4 (D_MAX = 32).
    thread float qbuf[D_MAX];
    thread float pbuf[D_MAX];
    thread float qold[D_MAX];
    thread float fbuf[D_MAX];
    thread float Areg[D_MAX * D_MAX];  // d*d <= 1024 floats per thread.

    // Zero-init everything (avoid uninitialised tail in vector ops).
    for (uint i = 0u; i < D_MAX; ++i) {
        qbuf[i] = 0.0f;
        pbuf[i] = 0.0f;
        fbuf[i] = 0.0f;
    }

    uint dpad = (d + 3u) & ~3u;
    uint dpad4 = dpad >> 2;

    // Load A into registers, padded layout: row i at offset i*dpad, tail zero.
    // Initialize the full Areg to zero for padded slots.
    for (uint i = 0u; i < d; ++i) {
        for (uint j = 0u; j < d; ++j) {
            Areg[i * dpad + j] = A[i * d + j];
        }
        for (uint j = d; j < dpad; ++j) {
            Areg[i * dpad + j] = 0.0f;
        }
    }

    uint base_counter = hmc_step_idx * (d + 1u);
    float eps_half = 0.5f * eps;

    // 1. Momentum p ~ N(0, I) via Box-Muller.
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

    // 2. Load q, save q_old.
    for (uint i = 0u; i < d; ++i) {
        float qi = q_in[chain_idx * d + i];
        qbuf[i] = qi;
        qold[i] = qi;
    }

    // Initial force = A q  and  U_old = 0.5 * q^T A q.
    float U_old = 0.0f;
    for (uint i = 0u; i < d; ++i) {
        thread const float4 *Arow = (thread const float4 *)(Areg + i * dpad);
        thread const float4 *qv   = (thread const float4 *)qbuf;
        float4 acc4 = float4(0.0f);
        for (uint j4 = 0u; j4 < dpad4; ++j4) {
            acc4 = fma(Arow[j4], qv[j4], acc4);
        }
        float acc = acc4.x + acc4.y + acc4.z + acc4.w;
        fbuf[i] = acc;
        U_old = fma(0.5f * qbuf[i], acc, U_old);
    }

    float K_old = 0.0f;
    for (uint i = 0u; i < d; ++i) {
        K_old = fma(0.5f * pbuf[i], pbuf[i], K_old);
    }

    // 3. Leapfrog. Initial half-kick.
    {
        thread float4 *pv = (thread float4 *)pbuf;
        thread const float4 *fv = (thread const float4 *)fbuf;
        float4 nh = float4(-eps_half);
        for (uint j4 = 0u; j4 < dpad4; ++j4) {
            pv[j4] = fma(nh, fv[j4], pv[j4]);
        }
    }

    for (uint l = 0u; l < L; ++l) {
        // Drift: q += eps * p
        {
            thread float4 *qv = (thread float4 *)qbuf;
            thread const float4 *pv = (thread const float4 *)pbuf;
            float4 e4 = float4(eps);
            for (uint j4 = 0u; j4 < dpad4; ++j4) {
                qv[j4] = fma(e4, pv[j4], qv[j4]);
            }
        }
        // Recompute force.
        for (uint i = 0u; i < d; ++i) {
            thread const float4 *Arow = (thread const float4 *)(Areg + i * dpad);
            thread const float4 *qv   = (thread const float4 *)qbuf;
            float4 acc4 = float4(0.0f);
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

    // 4. New Hamiltonian.
    float U_new = 0.0f;
    for (uint i = 0u; i < d; ++i) {
        U_new = fma(0.5f * qbuf[i], fbuf[i], U_new);
    }
    float K_new = 0.0f;
    for (uint i = 0u; i < d; ++i) {
        K_new = fma(0.5f * pbuf[i], pbuf[i], K_new);
    }

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