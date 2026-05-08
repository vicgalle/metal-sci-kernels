#include <metal_stdlib>
using namespace metal;

constant float TWO_PI = 6.28318530717958647692f;

inline uint log2_uint(uint n) {
    uint r = 0u;
    while ((1u << r) < n) ++r;
    return r;
}

inline uint bit_reverse_n(uint x, uint logN) {
    return reverse_bits(x) >> (32u - logN);
}

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// One TG of N threads runs a length-N FFT.
// `buf` must be preloaded with the bit-reversed input by the caller.
// On return, buf[i] holds the FFT result.
inline void line_fft(threadgroup float2 *buf,
                     threadgroup float2 *tw,
                     uint i, uint N, uint logN) {
    // Build twiddle table tw[k] = exp(-2πi * k / N) for k in [0, N/2).
    uint nh = N >> 1u;
    if (i < nh) {
        float ang = -TWO_PI * float(i) / float(N);
        float c;
        float s = sincos(ang, c);
        tw[i] = float2(c, s);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cooley-Tukey butterfly stages. N/2 active threads per stage.
    for (uint s = 1u; s <= logN; ++s) {
        uint m      = 1u << s;
        uint half_m = m >> 1u;
        if (i < nh) {
            uint group = i >> (s - 1u);          // i / half_m
            uint kk    = i & (half_m - 1u);
            uint base  = group << s;             // group * m
            uint tw_idx = kk << (logN - s);      // kk * (N / m)
            float2 w = tw[tw_idx];
            float2 a = buf[base + kk];
            float2 b = buf[base + half_m + kk];
            float2 t = cmul(w, b);
            buf[base + kk]          = a + t;
            buf[base + half_m + kk] = a - t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
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

    uint br = bit_reverse_n(i, logN);
    buf[i] = in_data[base_addr + br];

    line_fft(buf, tw, i, N, logN);

    out_data[base_addr + i] = buf[i];
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
    buf[j] = in_data[plane_base + br * N];

    line_fft(buf, tw, j, N, logN);

    out_data[plane_base + j * N] = buf[j];
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
    buf[k] = in_data[br * NN + col_base];

    line_fft(buf, tw, k, N, logN);

    out_data[k * NN + col_base] = buf[k];
}