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
    return ctz(n);
}

inline uint bit_reverse(uint x, uint logN) {
    return reverse_bits(x) >> (32u - logN);
}

// Maps indices to threadgroup memory banks evenly, eliminating power-of-two bank conflicts.
inline uint swizzle(uint idx) {
    return idx ^ (idx >> 4);
}

// Core 1D FFT loop. Computes one element 'x' for the thread 'tid'.
inline void compute_fft(uint tid, uint logN, thread float2& x, 
                        threadgroup float2* buf0, threadgroup float2* buf1) {
    // SIMD stages 1 to 5: entirely in registers using warp shuffles.
    for (uint s = 1; s <= 5; ++s) {
        if (s > logN) break;
        uint m = 1u << s;
        uint half_m = m >> 1u;
        uint kk = tid & (half_m - 1u);
        bool bottom = (tid & half_m) != 0;
        
        float angle = -TWO_PI * float(kk) / float(m);
        float c, s_;
        s_ = sincos(angle, c);
        float2 w = float2(c, s_);
        
        float2 v = simd_shuffle_xor(x, half_m);
        float2 bottom_val = bottom ? x : v;
        float2 top_val    = bottom ? v : x;
        
        float2 t = float2(w.x * bottom_val.x - w.y * bottom_val.y,
                          w.x * bottom_val.y + w.y * bottom_val.x);
                          
        x = bottom ? (top_val - t) : (top_val + t);
    }
    
    // Threadgroup stages 6 to logN: ping-pong shared memory.
    if (logN > 5) {
        buf0[swizzle(tid)] = x;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for (uint s = 6; s <= logN; ++s) {
            uint m = 1u << s;
            uint half_m = m >> 1u;
            uint kk = tid & (half_m - 1u);
            bool bottom = (tid & half_m) != 0;
            uint base = tid & ~(m - 1u);
            
            float angle = -TWO_PI * float(kk) / float(m);
            float c, s_;
            s_ = sincos(angle, c);
            float2 w = float2(c, s_);
            
            float2 u, v;
            if ((s & 1) == 0) { // s = 6, 8, 10
                u = buf0[swizzle(base + kk)];
                v = buf0[swizzle(base + half_m + kk)];
            } else {            // s = 7, 9, 11
                u = buf1[swizzle(base + kk)];
                v = buf1[swizzle(base + half_m + kk)];
            }
            
            float2 t = float2(w.x * v.x - w.y * v.y, w.x * v.y + w.y * v.x);
            x = bottom ? (u - t) : (u + t);
            
            // Skip write and barrier on the final stage to retain result in registers
            if (s != logN) {
                if ((s & 1) == 0) {
                    buf1[swizzle(tid)] = x;
                } else {
                    buf0[swizzle(tid)] = x;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}

kernel void fft3d_x(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) [[max_total_threads_per_threadgroup(1024)]]
{
    uint i = gid.x;
    uint line = gid.y;
    uint k = line / N;
    uint j = line - k * N;
    uint logN = log2_uint(N);

    threadgroup float2 buf0[1024];
    threadgroup float2 buf1[1024];

    uint src = (k * N + j) * N + i;
    buf0[swizzle(bit_reverse(i, logN))] = in_data[src];
    
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float2 x = buf0[swizzle(i)];
    compute_fft(i, logN, x, buf0, buf1);
    
    out_data[src] = x;
}

kernel void fft3d_y(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) [[max_total_threads_per_threadgroup(1024)]]
{
    uint j = gid.x;
    uint line = gid.y;
    uint k = line / N;
    uint i = line - k * N;
    uint logN = log2_uint(N);

    threadgroup float2 buf0[1024];
    threadgroup float2 buf1[1024];

    uint src = (k * N + j) * N + i;
    buf0[swizzle(bit_reverse(j, logN))] = in_data[src];
    
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float2 x = buf0[swizzle(j)];
    compute_fft(j, logN, x, buf0, buf1);
    
    out_data[src] = x;
}

kernel void fft3d_z(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) [[max_total_threads_per_threadgroup(1024)]]
{
    uint k = gid.x;
    uint line = gid.y;
    uint j = line / N;
    uint i = line - j * N;
    uint logN = log2_uint(N);

    threadgroup float2 buf0[1024];
    threadgroup float2 buf1[1024];

    uint src = (k * N + j) * N + i;
    buf0[swizzle(bit_reverse(k, logN))] = in_data[src];
    
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float2 x = buf0[swizzle(k)];
    compute_fft(k, logN, x, buf0, buf1);
    
    out_data[src] = x;
}
```

Result of previous attempt:
  COMPILE FAILED: Error Domain=MTLLibraryErrorDomain Code=3 "program_source:90:62: error: 'max_total_threads_per_threadgroup' attribute cannot be applied to types
                    uint3 gid [[thread_position_in_grid]]) [[max_total_threads_per_threadgroup(1024)]]
                                                             ^
program_source:115:62: error: 'max_total_threads_per_threadgroup' attribute cannot be applied to types
                    uint3 gid [[thread_position_in_grid]]) [[max_total_threads_per_threadgroup(1024)]]
                                                             ^
program_source:140:62: error: 'max_total_threads_per_threadgroup' attribute cannot be applied to types
                    uint3 gid [[thread_position_in_grid]]) [[max_total_threads_per_threadgroup(1024)]]
                                                             ^
" UserInfo={NSLocalizedDescription=program_source:90:62: error: 'max_total_threads_per_threadgroup' attribute cannot be applied to types
                    uint3 gid [[thread_position_in_grid]]) [[max_total_threads_per_threadgroup(1024)]]
                                                             ^
program_source:115:62: error: 'max_total_threads_per_threadgroup' attribute cannot be applied to types
                    uint3 gid [[thread_position_in_grid]]) [[max_total_threads_per_threadgroup(1024)]]
                                                             ^
program_source:140:62: error: 'max_total_threads_per_threadgroup' attribute cannot be applied to types
                    uint3 gid [[thread_position_in_grid]]) [[max_total_threads_per_threadgroup(1024)]]
                                                             ^
}

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
            32cube: correct, 0.12 ms, 26.2 GB/s (effective, 96 B/cell across 3 passes) (13.1% of 200 GB/s)
            64cube: correct, 0.57 ms, 44.5 GB/s (effective, 96 B/cell across 3 passes) (22.2% of 200 GB/s)
           128cube: correct, 2.21 ms, 90.9 GB/s (effective, 96 B/cell across 3 passes) (45.5% of 200 GB/s)
  score (gmean of fraction): 0.2366

## History

- iter  0: compile=OK | correct=True | score=0.2365849334564696
- iter  1: compile=FAIL | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
