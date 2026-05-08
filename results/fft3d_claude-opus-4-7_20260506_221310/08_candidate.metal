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

// Stockham auto-sort FFT, mixed radix-2/radix-4. Threadgroup of N threads,
// each thread owns one element. Twiddles tw[t] = exp(-2*pi*i * t / N) for
// t in [0, N/2) are precomputed cooperatively at entry.
//
// For radix-4 stages: each "active" thread (i < N/4) processes one
// 4-point butterfly, reading 4 inputs from `cur` and writing 4 outputs
// to `nxt` in Stockham layout. This halves the stage count vs radix-2.
//
// Returns pointer to the buffer holding the final result.
inline threadgroup float2* line_fft(threadgroup float2 *A,
                                    threadgroup float2 *B,
                                    threadgroup float2 *tw,
                                    uint i, uint N, uint logN) {
    uint Nh = N >> 1u;
    uint Nq = N >> 2u;

    // Cooperative twiddle precompute: tw[t] = exp(-2*pi*i * t / N).
    if (i < Nh) {
        float ang = -TWO_PI * float(i) / float(N);
        float c, s;
        s = sincos(ang, c);
        tw[i] = float2(c, s);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float2 *cur = A;
    threadgroup float2 *nxt = B;

    uint Ns = 1u;     // current sub-FFT size
    uint s  = 0u;     // log2(Ns)

    // If logN is odd, do one radix-2 stage first so remaining stages are radix-4.
    if ((logN & 1u) == 1u) {
        if (i < Nh) {
            // Ns = 1 here, so j = 0, k = i, w = 1.
            float2 x0 = cur[i];
            float2 x1 = cur[i + Nh];
            // Output base: k*(2*Ns) + j = 2*i + 0
            nxt[2u * i]      = x0 + x1;
            nxt[2u * i + 1u] = x0 - x1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float2 *tmp = cur; cur = nxt; nxt = tmp;
        Ns = 2u;
        s  = 1u;
    }

    // Radix-4 stages. Each thread (i < N/4) handles one 4-point butterfly.
    // Sub-FFT size goes Ns -> 4*Ns each stage.
    for (; s < logN; s += 2u) {
        if (i < Nq) {
            uint j = i & (Ns - 1u);    // index within sub-FFT
            uint k = i >> s;           // sub-FFT id (i / Ns)

            // Read 4 inputs from current Stockham layout.
            // In a radix-2 view, the four inputs are spaced by N/4.
            uint in0 = k * Ns + j;
            uint in1 = in0 + Nq;
            uint in2 = in1 + Nq;
            uint in3 = in2 + Nq;

            float2 x0 = cur[in0];
            float2 x1 = cur[in1];
            float2 x2 = cur[in2];
            float2 x3 = cur[in3];

            // Twiddles for radix-4: exponents j*(N/(4*Ns)) * {0,1,2,3}.
            // tw[j * step] where step = N >> (s+2) for power 1.
            uint step = N >> (s + 2u);
            uint t1 = j * step;
            uint t2 = t1 + t1;
            uint t3 = t2 + t1;

            // tw is defined only for indices in [0, N/2). t2, t3 may exceed.
            // tw[m + N/2] = -tw[m].  tw[m + N] = tw[m].
            float2 w1 = tw[t1 & (Nh - 1u)];
            if (t1 & Nh) w1 = -w1;
            float2 w2 = tw[t2 & (Nh - 1u)];
            if (t2 & Nh) w2 = -w2;
            float2 w3 = tw[t3 & (Nh - 1u)];
            if (t3 & Nh) w3 = -w3;

            float2 y1 = cmul(w1, x1);
            float2 y2 = cmul(w2, x2);
            float2 y3 = cmul(w3, x3);

            // Radix-4 DFT (forward, sign -):
            // Y0 = x0 + y1 + y2 + y3
            // Y1 = x0 - i*y1 - y2 + i*y3
            // Y2 = x0 - y1 + y2 - y3
            // Y3 = x0 + i*y1 - y2 - i*y3
            // Multiplication by -i: (a,b) -> (b,-a). By +i: (a,b) -> (-b, a).
            float2 a02p = x0 + y2;
            float2 a02m = x0 - y2;
            float2 a13p = y1 + y3;
            float2 a13m = y1 - y3;
            // i * a13m = (-a13m.y, a13m.x);  -i * a13m = (a13m.y, -a13m.x)
            float2 i_a13m_neg = float2( a13m.y, -a13m.x); // -i * a13m
            float2 i_a13m_pos = float2(-a13m.y,  a13m.x); // +i * a13m

            float2 Y0 = a02p + a13p;
            float2 Y1 = a02m + i_a13m_neg;
            float2 Y2 = a02p - a13p;
            float2 Y3 = a02m + i_a13m_pos;

            // Output base in Stockham layout: each sub-FFT of size Ns
            // expands to 4*Ns. Output index = k*(4*Ns) + j + r*Ns.
            uint outBase = (k << (s + 2u)) + j;
            nxt[outBase]            = Y0;
            nxt[outBase + Ns]       = Y1;
            nxt[outBase + 2u * Ns]  = Y2;
            nxt[outBase + 3u * Ns]  = Y3;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float2 *tmp = cur; cur = nxt; nxt = tmp;
        Ns <<= 2u;
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

    threadgroup float2 *res = line_fft(A, B, tw, i, N, logN);

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

    threadgroup float2 *res = line_fft(A, B, tw, j, N, logN);

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

    threadgroup float2 *res = line_fft(A, B, tw, k, N, logN);

    out_data[k * NN + col_base] = res[k];
}