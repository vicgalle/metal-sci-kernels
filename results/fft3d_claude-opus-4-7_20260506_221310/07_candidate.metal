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

// Stockham auto-sort radix-2 FFT with precomputed twiddles.
// `A` holds natural-order input on entry; result lands in either A or B
// depending on parity of logN — we return the pointer.
inline threadgroup float2* line_fft_stockham(threadgroup float2 *A,
                                             threadgroup float2 *B,
                                             threadgroup float2 *tw,
                                             uint i, uint N, uint logN) {
    uint Nh = N >> 1u;

    // Cooperative twiddle precompute: tw[t] = exp(-2*pi*i * t / N), t in [0, N/2).
    if (i < Nh) {
        float ang = -TWO_PI * float(i) / float(N);
        float c, s;
        s = sincos(ang, c);
        tw[i] = float2(c, s);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float2 *cur = A;
    threadgroup float2 *nxt = B;

    uint Ns = 1u; // current sub-FFT size = 2^s
    for (uint s = 0u; s < logN; ++s) {
        if (i < Nh) {
            uint j  = i & (Ns - 1u);   // index within sub-FFT
            uint k  = i >> s;          // sub-FFT id (i / Ns)
            uint in0 = k * Ns + j;
            uint in1 = in0 + Nh;

            // Twiddle: w = exp(-2*pi*i * j * (N/(2*Ns)) / N)
            //           = tw[j * (N >> (s+1))]
            uint twIdx = j << (logN - s - 1u);
            float2 w = tw[twIdx];

            float2 x0 = cur[in0];
            float2 x1 = cur[in1];
            float2 t  = cmul(w, x1);

            uint outBase = (k * Ns + j) + k * Ns; // = k*(2*Ns) + j
            nxt[outBase]      = x0 + t;
            nxt[outBase + Ns] = x0 - t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float2 *tmp = cur; cur = nxt; nxt = tmp;
        Ns <<= 1u;
    }
    return cur;
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

    threadgroup float2 A[1024];
    threadgroup float2 B[1024];
    threadgroup float2 tw[512];

    uint logN = log2_uint(N);
    uint base_addr = (k * N + j) * N;

    A[i] = in_data[base_addr + i];

    threadgroup float2 *res = line_fft_stockham(A, B, tw, i, N, logN);

    out_data[base_addr + i] = res[i];
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

    threadgroup float2 A[1024];
    threadgroup float2 B[1024];
    threadgroup float2 tw[512];

    uint logN = log2_uint(N);
    uint plane_base = k * N * N + i;

    A[j] = in_data[plane_base + j * N];

    threadgroup float2 *res = line_fft_stockham(A, B, tw, j, N, logN);

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

    threadgroup float2 A[1024];
    threadgroup float2 B[1024];
    threadgroup float2 tw[512];

    uint logN = log2_uint(N);
    uint NN = N * N;
    uint col_base = j * N + i;

    A[k] = in_data[k * NN + col_base];

    threadgroup float2 *res = line_fft_stockham(A, B, tw, k, N, logN);

    out_data[k * NN + col_base] = res[k];
}