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

// Stockham radix-4 (with optional final radix-2) length-N FFT.
// Threadgroup of N threads, ping-pong buffers A,B each of size N.
// Caller loads input into A[i] in natural order. Result returned in
// whichever buffer is "current" at the end; caller reads via the
// returned pointer. We achieve this by passing both and returning index.
inline threadgroup float2* line_fft_stockham_r4(threadgroup float2 *A,
                                                threadgroup float2 *B,
                                                uint i, uint N, uint logN) {
    threadgroup float2 *cur = A;
    threadgroup float2 *nxt = B;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint Ns = 1u;          // sub-FFT size processed so far
    uint stages_left = logN;

    // Radix-4 stages
    while (stages_left >= 2u) {
        uint q = N >> 2u;              // number of butterflies (= N/4)
        if (i < q) {
            // Decompose i into (j, k): j in [0,Ns), k in [0, N/(4*Ns))
            uint j = i & (Ns - 1u);
            uint k = i >> uint(log2_uint(Ns));   // safe; Ns is pow2
            // Indices of the 4 inputs (stride q in cur)
            uint idx0 = k * Ns + j;
            uint i0 = idx0;
            uint i1 = idx0 + q;
            uint i2 = idx0 + 2u * q;
            uint i3 = idx0 + 3u * q;

            float2 x0 = cur[i0];
            float2 x1 = cur[i1];
            float2 x2 = cur[i2];
            float2 x3 = cur[i3];

            // Twiddle base angle
            float ang = -TWO_PI * float(j) / float(4u * Ns);
            float c1, s1, c2, s2, c3, s3;
            s1 = sincos(ang,        c1);
            s2 = sincos(2.0f * ang, c2);
            s3 = sincos(3.0f * ang, c3);

            float2 w1 = float2(c1, s1);
            float2 w2 = float2(c2, s2);
            float2 w3 = float2(c3, s3);

            float2 y1 = cmul(w1, x1);
            float2 y2 = cmul(w2, x2);
            float2 y3 = cmul(w3, x3);

            // Radix-4 butterfly (forward, -i factor)
            float2 t0 = x0 + y2;
            float2 t1 = x0 - y2;
            float2 t2 = y1 + y3;
            float2 t3 = y1 - y3;
            // multiply t3 by -i (forward FFT): (a,b)*(-i) = (b,-a)
            float2 t3mi = float2(t3.y, -t3.x);

            float2 r0 = t0 + t2;
            float2 r1 = t1 + t3mi;
            float2 r2 = t0 - t2;
            float2 r3 = t1 - t3mi;

            // Output indexing for Stockham: position = k*(4*Ns) + j + m*Ns
            uint outBase = k * (Ns << 2u) + j;
            nxt[outBase + 0u * Ns] = r0;
            nxt[outBase + 1u * Ns] = r1;
            nxt[outBase + 2u * Ns] = r2;
            nxt[outBase + 3u * Ns] = r3;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // swap
        threadgroup float2 *tmp = cur; cur = nxt; nxt = tmp;
        Ns <<= 2u;
        stages_left -= 2u;
    }

    // Final radix-2 stage if logN is odd
    if (stages_left == 1u) {
        uint q = N >> 1u;
        if (i < q) {
            uint j = i & (Ns - 1u);
            uint k = i >> uint(log2_uint(Ns));
            uint idx0 = k * Ns + j;
            uint i0 = idx0;
            uint i1 = idx0 + q;

            float2 x0 = cur[i0];
            float2 x1 = cur[i1];

            float ang = -TWO_PI * float(j) / float(2u * Ns);
            float c, s;
            s = sincos(ang, c);
            float2 w = float2(c, s);
            float2 y1 = cmul(w, x1);

            uint outBase = k * (Ns << 1u) + j;
            nxt[outBase]       = x0 + y1;
            nxt[outBase + Ns]  = x0 - y1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float2 *tmp = cur; cur = nxt; nxt = tmp;
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

    uint logN = log2_uint(N);
    uint base_addr = (k * N + j) * N;

    A[i] = in_data[base_addr + i];

    threadgroup float2 *res = line_fft_stockham_r4(A, B, i, N, logN);

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

    uint logN = log2_uint(N);
    uint plane_base = k * N * N + i;

    A[j] = in_data[plane_base + j * N];

    threadgroup float2 *res = line_fft_stockham_r4(A, B, j, N, logN);

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

    uint logN = log2_uint(N);
    uint NN = N * N;
    uint col_base = j * N + i;

    A[k] = in_data[k * NN + col_base];

    threadgroup float2 *res = line_fft_stockham_r4(A, B, k, N, logN);

    out_data[k * NN + col_base] = res[k];
}