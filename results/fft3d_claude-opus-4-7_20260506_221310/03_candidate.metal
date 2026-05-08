#include <metal_stdlib>
using namespace metal;

constant float TWO_PI = 6.28318530717958647692f;

inline uint log2_uint(uint n) {
    uint r = 0u;
    while ((1u << r) < n) ++r;
    return r;
}

inline uint bit_reverse_n(uint x, uint logN) {
    // reverse_bits gives 32-bit reversal; shift down to logN bits.
    return reverse_bits(x) >> (32u - logN);
}

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Core: one TG of N threads runs a length-N FFT.
// `val` holds this thread's element after bit-reversed load (in register).
// First 5 stages (span 2..32) run via simdgroup shuffles; remaining stages
// use threadgroup memory with twiddle table cached in `tw`.
inline float2 line_fft_core(float2 val,
                            threadgroup float2 *buf,
                            threadgroup float2 *tw,
                            uint i, uint N, uint logN) {
    uint lane = i & 31u;

    // Build twiddle table tw[k] = exp(-2πi * k / N) for k in [0, N/2).
    // Each thread fills one or more entries.
    uint nh = N >> 1u;
    for (uint k = i; k < nh; k += N) {
        float ang = -TWO_PI * float(k) / float(N);
        float c, s;
        s = sincos(ang, c);
        tw[k] = float2(c, s);
    }

    // ---------- Stages 1..min(5,logN): simdgroup shuffle butterflies ----------
    uint sg_stages = min(5u, logN);

    for (uint s = 1u; s <= sg_stages; ++s) {
        uint m      = 1u << s;
        uint half_m = m >> 1u;
        uint partner_lane = lane ^ half_m;
        float2 partner = simd_shuffle_xor(val, half_m);

        // kk = position within the butterfly group of size m.
        uint kk = lane & (half_m - 1u);
        // Twiddle stride: N / m.
        uint tw_idx = kk * (N >> s);
        float2 w = tw[tw_idx];

        // If we're the lower half (lane bit `half_m` is 0): u = val, t = w * partner
        //                                          result u + t
        // Else (upper half):                                t = val, u = partner
        //                                          result u - t (with w applied to t)
        bool lower = (lane & half_m) == 0u;
        float2 u = lower ? val     : partner;
        float2 v = lower ? partner : val;
        float2 t = cmul(w, v);
        val = lower ? (u + t) : (u - t);
    }

    // If logN <= 5 we're done; just write to threadgroup so caller can read.
    if (logN <= sg_stages) {
        return val;
    }

    // Spill to threadgroup memory for cross-simd stages.
    buf[i] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---------- Stages sg_stages+1 .. logN: threadgroup memory butterflies ----------
    for (uint s = sg_stages + 1u; s <= logN; ++s) {
        uint m      = 1u << s;
        uint half_m = m >> 1u;
        if (i < nh) {
            uint group = i >> (s - 1u);          // i / half_m
            uint kk    = i & (half_m - 1u);
            uint base  = group << s;             // group * m
            uint tw_idx = kk * (N >> s);
            float2 w = tw[tw_idx];
            float2 a = buf[base + kk];
            float2 b = buf[base + half_m + kk];
            float2 t = cmul(w, b);
            buf[base + kk]          = a + t;
            buf[base + half_m + kk] = a - t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    return buf[i];
}

[[max_total_threads_per_threadgroup(1024)]]
kernel void fft3d_x(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint line = gid.y;
    uint k = line / N;
    uint j = line - k * N;

    threadgroup float2 buf[1024];
    threadgroup float2 tw[512];

    uint logN = log2_uint(N);
    uint base_addr = (k * N + j) * N;

    // Bit-reversed cooperative load directly into register; permutation
    // is realized by shuffling within the simd / via threadgroup buf.
    uint br = bit_reverse_n(i, logN);
    float2 val = in_data[base_addr + br];

    float2 res = line_fft_core(val, buf, tw, i, N, logN);

    out_data[base_addr + i] = res;
}

[[max_total_threads_per_threadgroup(1024)]]
kernel void fft3d_y(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint j = gid.x;
    uint line = gid.y;
    uint k = line / N;
    uint i = line - k * N;

    threadgroup float2 buf[1024];
    threadgroup float2 tw[512];

    uint logN = log2_uint(N);
    uint plane_base = k * N * N + i;

    uint br = bit_reverse_n(j, logN);
    float2 val = in_data[plane_base + br * N];

    float2 res = line_fft_core(val, buf, tw, j, N, logN);

    out_data[plane_base + j * N] = res;
}

[[max_total_threads_per_threadgroup(1024)]]
kernel void fft3d_z(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint k = gid.x;
    uint line = gid.y;
    uint j = line / N;
    uint i = line - j * N;

    threadgroup float2 buf[1024];
    threadgroup float2 tw[512];

    uint logN = log2_uint(N);
    uint NN = N * N;
    uint col_base = j * N + i;

    uint br = bit_reverse_n(k, logN);
    float2 val = in_data[br * NN + col_base];

    float2 res = line_fft_core(val, buf, tw, k, N, logN);

    out_data[k * NN + col_base] = res;
}