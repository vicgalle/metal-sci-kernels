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
```

Result of previous attempt:
            32cube: correct, 0.30 ms, 10.4 GB/s (effective, 96 B/cell across 3 passes) (5.2% of 200 GB/s)
            64cube: correct, 1.29 ms, 19.6 GB/s (effective, 96 B/cell across 3 passes) (9.8% of 200 GB/s)
           128cube: correct, 3.35 ms, 60.2 GB/s (effective, 96 B/cell across 3 passes) (30.1% of 200 GB/s)
  score (gmean of fraction): 0.1153

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

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
