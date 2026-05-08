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
```

Result of previous attempt:
            32cube: correct, 0.35 ms, 8.9 GB/s (effective, 96 B/cell across 3 passes) (4.5% of 200 GB/s)
            64cube: correct, 1.28 ms, 19.7 GB/s (effective, 96 B/cell across 3 passes) (9.9% of 200 GB/s)
           128cube: correct, 3.54 ms, 56.8 GB/s (effective, 96 B/cell across 3 passes) (28.4% of 200 GB/s)
  score (gmean of fraction): 0.1078

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

- iter  1: compile=FAIL | correct=False | score=N/A
- iter  2: compile=OK | correct=True | score=0.11529641432202599
- iter  3: compile=OK | correct=False | score=N/A
- iter  4: compile=OK | correct=True | score=0.14409712412730596
- iter  5: compile=OK | correct=True | score=0.11274212668728693
- iter  6: compile=FAIL | correct=False | score=N/A
- iter  7: compile=OK | correct=True | score=0.09875536218312887
- iter  8: compile=OK | correct=True | score=0.107820724987393

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
