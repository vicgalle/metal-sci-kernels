#include <metal_stdlib>
using namespace metal;

constant float TWO_PI = 6.28318530717958647692f;

inline uint log2_uint(uint n) {
    uint r = 0u;
    while ((1u << r) < n) ++r;
    return r;
}

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Stockham auto-sort FFT. Each thread (id `i` in [0,N)) computes one
// butterfly per stage when i < N/2. Ping-pongs between bufA and bufB.
// Returns a pointer to whichever buffer holds the final natural-order result.
inline threadgroup float2 *stockham_run(threadgroup float2 *bufA,
                                        threadgroup float2 *bufB,
                                        uint i, uint N, uint logN) {
    threadgroup float2 *src = bufA;
    threadgroup float2 *dst = bufB;
    uint nh = N >> 1u;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 0u; s < logN; ++s) {
        uint m = 1u << s;            // input subFFT size
        if (i < nh) {
            uint kk    = i & (m - 1u);
            uint group = i >> s;
            uint in0   = group * m + kk;
            uint in1   = in0 + nh;
            uint out0  = (group << (s + 1u)) + kk;
            uint out1  = out0 + m;

            float angle = -TWO_PI * float(kk) / float(m << 1u);
            float c;
            float sn = sincos(angle, c);
            float2 w = float2(c, sn);
            float2 u = src[in0];
            float2 v = cmul(w, src[in1]);
            dst[out0] = u + v;
            dst[out1] = u - v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float2 *tmp = src; src = dst; dst = tmp;
    }
    return src;
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

    threadgroup float2 bufA[1024];
    threadgroup float2 bufB[1024];

    uint logN = log2_uint(N);
    uint base = (k * N + j) * N;

    bufA[i] = in_data[base + i];

    threadgroup float2 *res = stockham_run(bufA, bufB, i, N, logN);

    out_data[base + i] = res[i];
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

    threadgroup float2 bufA[1024];
    threadgroup float2 bufB[1024];

    uint logN = log2_uint(N);
    uint plane_base = k * N * N + i;

    bufA[j] = in_data[plane_base + j * N];

    threadgroup float2 *res = stockham_run(bufA, bufB, j, N, logN);

    out_data[plane_base + j * N] = res[j];
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

    threadgroup float2 bufA[1024];
    threadgroup float2 bufB[1024];

    uint logN = log2_uint(N);
    uint NN = N * N;
    uint col_base = j * N + i;

    bufA[k] = in_data[k * NN + col_base];

    threadgroup float2 *res = stockham_run(bufA, bufB, k, N, logN);

    out_data[k * NN + col_base] = res[k];
}