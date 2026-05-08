## Task: fft3d

3D complex-to-complex forward FFT, fp32, on a power-of-two cube of side N. Convention: forward, unnormalized — 
  Y[k,j,i] = sum_{kk,jj,ii} X[kk,jj,ii] * exp(-2πi (k·kk + j·jj + i·ii) / N)
(matches numpy.fft.fftn with norm='backward').

Storage is row-major float2[NZ][NY][NX] with NX=NY=NZ=N. Linear index of element (i,j,k) is ((k·N + j)·N + i); float2 is (real, imag) and is the buffer element type. The host calls three separate kernels — fft3d_x, fft3d_y, fft3d_z — in that order, ping-ponging between two device buffers (so the final 3D FFT result lands in the second buffer). Each kernel does one 1D length-N FFT per threadgroup; the FFT axis is fixed by the kernel name and its index decoding.

Because the three axes are orthogonal, the FFTs commute — the result is invariant to the order x→y→z vs any other order, but the host fixes the order x→y→z and the kernel names must match. The optimization surface is dominated by data movement: bit-reversal vs Stockham auto-sort, twiddle caching, simdgroup-shuffle butterflies, and threadgroup-memory bank-conflict avoidance are all on the table.

## Required kernel signature(s)

```
kernel void fft3d_x(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]);
kernel void fft3d_y(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]);
kernel void fft3d_z(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]);

Dispatch geometry (identical for all three kernels, host-fixed):
  threadsPerGrid        = (N, N*N, 1)
  threadsPerThreadgroup = (N, 1,   1)
So each TG of N threads owns exactly one length-N line: gid.x is the position along the FFT axis (= thread_position_in_threadgroup.x) and gid.y indexes the (N×N) plane of lines orthogonal to that axis. Index decoding per kernel:
  fft3d_x: i = gid.x; k = gid.y / N; j = gid.y - k*N
  fft3d_y: j = gid.x; k = gid.y / N; i = gid.y - k*N
  fft3d_z: k = gid.x; j = gid.y / N; i = gid.y - j*N
Each TG must produce the full FFT of its line in out_data; the host runs the three kernels back-to-back in one command buffer and ping-pongs the buffers, so out_data of one pass is the in_data of the next.

If you cap the threadgroup with [[max_total_threads_per_threadgroup(N)]], place the attribute on the kernel declaration line itself (not as a free-standing statement), and remember the host's TG width along x is N (≤ 1024 on M-series). Your tile / shared-memory layout MUST match the dispatched (N, 1, 1) TG geometry: the host will not split or reshape the dispatch to fit a different tile.
```

## Your previous attempt

```metal
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
```

Result of previous attempt:
            32cube: correct, 0.34 ms, 9.3 GB/s (effective, 96 B/cell across 3 passes) (4.6% of 200 GB/s)
            64cube: correct, 1.33 ms, 18.9 GB/s (effective, 96 B/cell across 3 passes) (9.5% of 200 GB/s)
           128cube: correct, 3.08 ms, 65.4 GB/s (effective, 96 B/cell across 3 passes) (32.7% of 200 GB/s)
  score (gmean of fraction): 0.1127

## Current best (incumbent)

```metal
// Naive seed for 3D complex-to-complex forward FFT (fp32, power-of-two cube N).
//
// Three named kernels (one per axis) — the host calls them in order
// fft3d_x -> fft3d_y -> fft3d_z, ping-ponging between two buffers. Each
// kernel is one length-N 1D FFT done by one threadgroup of N threads in
// shared memory: cooperative load with bit-reversal permutation, then
// log2(N) Cooley-Tukey butterfly stages with sin/cos twiddles computed
// on the fly. Stride-1 reads/writes for x; strided N (or N*N) for y, z.
//
// Convention: forward, unnormalized. Y_k = sum_n X_n * exp(-2πi k n / N).
// Matches numpy.fft.fftn (norm="backward", default).
//
// Dispatch (host-provided, identical across all three kernels):
//   threadsPerGrid       = (N, N*N, 1)
//   threadsPerThreadgroup= (N, 1,  1)
// So gid.x ∈ [0,N) is the position along the FFT axis (= lid.x), and
// gid.y ∈ [0,N²) is the line index within the (N×N) plane orthogonal to
// the FFT axis.
//
// Storage: row-major float2[NZ][NY][NX] with NX=NY=NZ=N. Linear index of
// element (i,j,k) is ((k*N + j)*N + i). float2 = (real, imag).

#include <metal_stdlib>
using namespace metal;

constant float TWO_PI = 6.28318530717958647692f;

inline uint log2_uint(uint n) {
    uint r = 0u;
    while ((1u << r) < n) ++r;
    return r;
}

inline uint bit_reverse(uint x, uint logN) {
    uint r = 0u;
    for (uint b = 0u; b < logN; ++b) {
        r = (r << 1) | (x & 1u);
        x >>= 1u;
    }
    return r;
}

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// One TG of N threads cooperates on one length-N FFT in `buf`. The caller
// is responsible for loading `buf` with the bit-reversed input and
// writing back the result; this routine performs only the butterfly
// stages with a barrier between each.
inline void line_fft_butterflies(threadgroup float2 *buf,
                                 uint i, uint N, uint logN) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 1u; s <= logN; ++s) {
        uint m      = 1u << s;       // current butterfly span
        uint half_m = m >> 1u;
        if (i < (N >> 1u)) {
            uint group = i / half_m;
            uint kk    = i - group * half_m;
            uint base  = group * m;
            float angle = -TWO_PI * float(kk) / float(m);
            float c = cos(angle), s_ = sin(angle);
            float2 w = float2(c, s_);
            float2 t = cmul(w, buf[base + half_m + kk]);
            float2 u = buf[base + kk];
            buf[base + kk]          = u + t;
            buf[base + half_m + kk] = u - t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// FFT along x (stride 1). Line index gid.y decodes (j, k); element gid.x = i.
kernel void fft3d_x(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint line = gid.y;
    uint k = line / N;
    uint j = line - k * N;

    threadgroup float2 buf[1024];
    uint logN = log2_uint(N);

    if (i < N) {
        uint src = (k * N + j) * N + i;
        buf[bit_reverse(i, logN)] = in_data[src];
    }
    line_fft_butterflies(buf, i, N, logN);
    if (i < N) {
        uint dst = (k * N + j) * N + i;
        out_data[dst] = buf[i];
    }
}

// FFT along y (stride N). Line index gid.y decodes (i, k); element gid.x = j.
kernel void fft3d_y(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint j = gid.x;
    uint line = gid.y;
    uint k = line / N;
    uint i = line - k * N;

    threadgroup float2 buf[1024];
    uint logN = log2_uint(N);

    if (j < N) {
        uint src = (k * N + j) * N + i;
        buf[bit_reverse(j, logN)] = in_data[src];
    }
    line_fft_butterflies(buf, j, N, logN);
    if (j < N) {
        uint dst = (k * N + j) * N + i;
        out_data[dst] = buf[j];
    }
}

// FFT along z (stride N*N). Line index gid.y decodes (i, j); element gid.x = k.
kernel void fft3d_z(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint k = gid.x;
    uint line = gid.y;
    uint j = line / N;
    uint i = line - j * N;

    threadgroup float2 buf[1024];
    uint logN = log2_uint(N);

    if (k < N) {
        uint src = (k * N + j) * N + i;
        buf[bit_reverse(k, logN)] = in_data[src];
    }
    line_fft_butterflies(buf, k, N, logN);
    if (k < N) {
        uint dst = (k * N + j) * N + i;
        out_data[dst] = buf[k];
    }
}
```

Incumbent result:
            32cube: correct, 0.22 ms, 14.0 GB/s (effective, 96 B/cell across 3 passes) (7.0% of 200 GB/s)
            64cube: correct, 0.93 ms, 27.1 GB/s (effective, 96 B/cell across 3 passes) (13.5% of 200 GB/s)
           128cube: correct, 2.26 ms, 89.2 GB/s (effective, 96 B/cell across 3 passes) (44.6% of 200 GB/s)
  score (gmean of fraction): 0.1619

## History

- iter  0: compile=OK | correct=True | score=0.16186407416071655
- iter  1: compile=FAIL | correct=False | score=N/A
- iter  2: compile=OK | correct=True | score=0.11529641432202599
- iter  3: compile=OK | correct=False | score=N/A
- iter  4: compile=OK | correct=True | score=0.14409712412730596
- iter  5: compile=OK | correct=True | score=0.11274212668728693

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
