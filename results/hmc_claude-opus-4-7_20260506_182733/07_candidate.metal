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

// ===========================================================================
// Templated worker with state in threadgroup memory laid out [D][TG_W].
// Each thread (chain) owns column `lane` across all D rows.
// Stride between successive elements of a per-thread vector is TG_W.
// ===========================================================================
template <uint D, uint TG_W>
inline void hmc_run_tg(uint chain_idx,
                       uint lane,
                       device const float *q_in,
                       device       float *q_out,
                       device       uint  *accept_cnt,
                       threadgroup const float *Atile,        // row-major D*D
                       threadgroup float *qmem,                // [D][TG_W]
                       threadgroup float *pmem,                // [D][TG_W]
                       threadgroup float *fmem,                // [D][TG_W]
                       threadgroup float *qoldmem,             // [D][TG_W]
                       uint L, float eps,
                       uint hmc_step_idx, uint seed)
{
    uint base_counter = hmc_step_idx * (D + 1u);
    float eps_half = 0.5f * eps;
    const float TWO_PI = 6.2831853071795864f;

    // ---- 1) Sample momentum p ~ N(0, I) via Box-Muller ----
    #pragma unroll
    for (uint i = 0u; i < D; i += 2u) {
        uint b1 = rand_u32(seed, chain_idx, base_counter + i);
        uint b2 = rand_u32(seed, chain_idx, base_counter + i + 1u);
        float u1 = u01_from_bits(b1);
        float u2 = u01_from_bits(b2);
        u1 = max(u1, 1.0e-7f);
        float r = sqrt(-2.0f * log(u1));
        float angle = TWO_PI * u2;
        float c;
        float s = sincos(angle, c);
        pmem[i * TG_W + lane]        = r * c;
        pmem[(i + 1u) * TG_W + lane] = r * s;
    }

    // ---- 2) Load q, save q_old ----
    device const float *qin = q_in + chain_idx * D;
    #pragma unroll
    for (uint i = 0u; i < D; ++i) {
        float qi = qin[i];
        qmem[i * TG_W + lane]    = qi;
        qoldmem[i * TG_W + lane] = qi;
    }

    // matvec f = A q
    #pragma unroll
    for (uint i = 0u; i < D; ++i) {
        threadgroup const float *Arow = Atile + i * D;
        float acc = 0.0f;
        #pragma unroll
        for (uint j = 0u; j < D; ++j) {
            acc = fma(Arow[j], qmem[j * TG_W + lane], acc);
        }
        fmem[i * TG_W + lane] = acc;
    }

    // U_old, K_old
    float U_old = 0.0f;
    float K_old = 0.0f;
    #pragma unroll
    for (uint i = 0u; i < D; ++i) {
        float qi = qmem[i * TG_W + lane];
        float fi = fmem[i * TG_W + lane];
        float pi = pmem[i * TG_W + lane];
        U_old = fma(qi, fi, U_old);
        K_old = fma(pi, pi, K_old);
    }
    U_old *= 0.5f;
    K_old *= 0.5f;

    // ---- 3) Leapfrog ----
    // initial half-kick
    #pragma unroll
    for (uint i = 0u; i < D; ++i) {
        float pi = pmem[i * TG_W + lane];
        float fi = fmem[i * TG_W + lane];
        pmem[i * TG_W + lane] = fma(-eps_half, fi, pi);
    }

    for (uint l = 0u; l < L; ++l) {
        // drift: q += eps * p
        #pragma unroll
        for (uint i = 0u; i < D; ++i) {
            float qi = qmem[i * TG_W + lane];
            float pi = pmem[i * TG_W + lane];
            qmem[i * TG_W + lane] = fma(eps, pi, qi);
        }
        // f = A q
        #pragma unroll
        for (uint i = 0u; i < D; ++i) {
            threadgroup const float *Arow = Atile + i * D;
            float acc = 0.0f;
            #pragma unroll
            for (uint j = 0u; j < D; ++j) {
                acc = fma(Arow[j], qmem[j * TG_W + lane], acc);
            }
            fmem[i * TG_W + lane] = acc;
        }
        // kick
        float scale = (l + 1u == L) ? -eps_half : -eps;
        #pragma unroll
        for (uint i = 0u; i < D; ++i) {
            float pi = pmem[i * TG_W + lane];
            float fi = fmem[i * TG_W + lane];
            pmem[i * TG_W + lane] = fma(scale, fi, pi);
        }
    }

    // ---- 4) U_new, K_new ----
    float U_new = 0.0f;
    float K_new = 0.0f;
    #pragma unroll
    for (uint i = 0u; i < D; ++i) {
        float qi = qmem[i * TG_W + lane];
        float fi = fmem[i * TG_W + lane];
        float pi = pmem[i * TG_W + lane];
        U_new = fma(qi, fi, U_new);
        K_new = fma(pi, pi, K_new);
    }
    U_new *= 0.5f;
    K_new *= 0.5f;

    // ---- 5) Accept / reject ----
    float dH = (U_new + K_new) - (U_old + K_old);
    uint b_acc = rand_u32(seed, chain_idx, base_counter + D);
    float u_acc = u01_from_bits(b_acc);
    float log_u = log(max(u_acc, 1.0e-30f));
    bool accept = isfinite(dH) && (log_u < -dH);

    device float *qout = q_out + chain_idx * D;
    #pragma unroll
    for (uint i = 0u; i < D; ++i) {
        float qi    = qmem[i * TG_W + lane];
        float qoldi = qoldmem[i * TG_W + lane];
        qout[i] = accept ? qi : qoldi;
    }
    if (accept) {
        accept_cnt[chain_idx] = accept_cnt[chain_idx] + 1u;
    }
}

// ===========================================================================
// Entry kernels — one per d. Threadgroup width chosen to fit tg memory budget.
// ===========================================================================

// d=8:  TG_W=128.  per-chain tg floats = 4*8 = 32 floats.  total = 4*8*128 = 4096 floats = 16KB. + A (256B) = ~16KB.
[[max_total_threads_per_threadgroup(128)]]
kernel void hmc_step_d8(device const float *q_in        [[buffer(0)]],
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
                        uint tg_size   [[threads_per_threadgroup]]);

// We instead expose ONE entry point matching the required signature and
// switch internally on d.
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
    // Sized for worst case d=32, TG_W=64: A=4KB, state=4*32*64=32KB floats=... too big.
    // Instead: per-d sizing using #if-style template specialization at runtime.
    //
    // We allocate the maximum we use:
    //   d=8,  TG_W=128 -> A=256, state=4*8*128*4 = 16KB
    //   d=16, TG_W=64  -> A=1KB, state=4*16*64*4 = 16KB
    //   d=32, TG_W=32  -> A=4KB, state=4*32*32*4 = 16KB
    // We pick the threadgroup width on the host side normally; here we just
    // cap based on d using compile-time template instantiations and runtime
    // dispatch. Total threadgroup memory needed: A_max=4KB + state=16KB = 20KB.

    threadgroup float Atile[32u * 32u];           // up to 4 KB
    threadgroup float qmem   [32u * 128u];        // 16 KB
    threadgroup float pmem   [32u * 128u];        // 16 KB  -- but tg mem is only ~32KB total
    // The above two would exceed budget if all simultaneously live for d=32.
    // Reality: only one (D, TG_W) combo runs per dispatch. Compiler still
    // allocates the union — fine because total static tg alloc is bounded.
    // To stay under 32KB we instead inline three small variants below.
    // (We'll fall back to a single shared buffer scheme.)

    // ---- Cooperative load of A ----
    uint dd = d * d;
    for (uint k = tid_in_tg; k < dd; k += tg_size) {
        Atile[k] = A[k];
    }

    // Use qmem region as a single "state" arena split into 4 sections
    // of size D * TG_W each. We treat pmem as the second slot of the same
    // arena — but to stay correct, we layout offsets explicitly.
    // Layout in qmem buffer: [q | p | f | qold], each D*TG_W.
    threadgroup float *state = qmem;  // up to 32*128 = 4096 floats = 16 KB
    // We need 4*D*TG_W floats <= 4096 -> D*TG_W <= 1024.
    // d=8 :TG_W<=128; d=16:TG_W<=64; d=32:TG_W<=32. Good.

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (chain_idx >= K) return;

    uint lane = tid_in_tg;

    if (d == 8u) {
        const uint D = 8u;
        const uint TG_W = 128u;
        threadgroup float *qm    = state + 0u * D * TG_W;
        threadgroup float *pm    = state + 1u * D * TG_W;
        threadgroup float *fm    = state + 2u * D * TG_W;
        threadgroup float *qoldm = state + 3u * D * TG_W;
        hmc_run_tg<D, TG_W>(chain_idx, lane, q_in, q_out, accept_cnt,
                            Atile, qm, pm, fm, qoldm,
                            L, eps, hmc_step_idx, seed);
    } else if (d == 16u) {
        const uint D = 16u;
        const uint TG_W = 64u;
        threadgroup float *qm    = state + 0u * D * TG_W;
        threadgroup float *pm    = state + 1u * D * TG_W;
        threadgroup float *fm    = state + 2u * D * TG_W;
        threadgroup float *qoldm = state + 3u * D * TG_W;
        hmc_run_tg<D, TG_W>(chain_idx, lane, q_in, q_out, accept_cnt,
                            Atile, qm, pm, fm, qoldm,
                            L, eps, hmc_step_idx, seed);
    } else {
        const uint D = 32u;
        const uint TG_W = 32u;
        threadgroup float *qm    = state + 0u * D * TG_W;
        threadgroup float *pm    = state + 1u * D * TG_W;
        threadgroup float *fm    = state + 2u * D * TG_W;
        threadgroup float *qoldm = state + 3u * D * TG_W;
        hmc_run_tg<D, TG_W>(chain_idx, lane, q_in, q_out, accept_cnt,
                            Atile, qm, pm, fm, qoldm,
                            L, eps, hmc_step_idx, seed);
    }
    // pmem reference to silence unused warning (it is unused in this layout).
    (void)pmem;
}