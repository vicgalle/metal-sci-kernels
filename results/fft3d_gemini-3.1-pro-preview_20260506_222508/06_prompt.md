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

// Stockham DIF butterfly network.
// Radix-2 Decimation-In-Frequency algorithm that inherently avoids bit-reversal.
// By ping-ponging between buf0 and buf1, the final output remains perfectly in-order.
inline void stockham_dif_butterflies(
    threadgroup float *buf0_re,
    threadgroup float *buf0_im,
    threadgroup float *buf1_re,
    threadgroup float *buf1_im,
    uint idx, uint N, uint logN)
{
    threadgroup float *src_re = buf0_re;
    threadgroup float *src_im = buf0_im;
    threadgroup float *dst_re = buf1_re;
    threadgroup float *dst_im = buf1_im;
    
    uint half_N = N >> 1u;
    for (uint s = 1u; s <= logN; ++s) {
        uint log2_half_m = logN - s;
        uint half_m = 1u << log2_half_m;
        
        uint log2_m = log2_half_m + 1u;
        // Fast exact 1.0f / m using IEEE-754 exponent injection
        float inv_m = as_type<float>((127u - log2_m) << 23u);
        float angle_step = -TWO_PI * inv_m;
        
        // Only the lower N/2 threads compute butterflies; the upper half just synchronizes.
        if (idx < half_N) {
            uint j = idx >> log2_half_m;
            uint k = idx & (half_m - 1u);
            
            uint in_idx0 = (j << log2_m) | k;
            uint in_idx1 = in_idx0 | half_m;
            
            float u_r = src_re[in_idx0];
            float u_i = src_im[in_idx0];
            float v_r = src_re[in_idx1];
            float v_i = src_im[in_idx1];
            
            float sum_r = u_r + v_r;
            float sum_i = u_i + v_i;
            float diff_r = u_r - v_r;
            float diff_i = u_i - v_i;
            
            float angle = angle_step * float(k);
            float c, s_;
            s_ = sincos(angle, c);
            
            float t_r = c * diff_r - s_ * diff_i;
            float t_i = c * diff_i + s_ * diff_r;
            
            // Stockham DIF writes are perfectly contiguous!
            dst_re[idx]          = sum_r;
            dst_im[idx]          = sum_i;
            dst_re[idx + half_N] = t_r;
            dst_im[idx + half_N] = t_i;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Ping-pong buffers
        threadgroup float *tmp_r = src_re; src_re = dst_re; dst_re = tmp_r;
        threadgroup float *tmp_i = src_im; src_im = dst_im; dst_im = tmp_i;
    }
}

[[max_total_threads_per_threadgroup(1024)]]
kernel void fft3d_x(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) 
{
    uint idx = gid.x;
    uint line = gid.y;
    uint logN = ctz(N);
    
    threadgroup float buf0_re[1024];
    threadgroup float buf0_im[1024];
    threadgroup float buf1_re[1024];
    threadgroup float buf1_im[1024];
    
    // Contiguous access allows perfect float4 vectorization (doubling transfer rate)
    if (idx < (N >> 1u)) {
        uint src_vec = (line << (logN - 1u)) | idx;
        float4 val = ((device const float4*)in_data)[src_vec];
        uint i0 = idx << 1u;
        uint i1 = i0 | 1u;
        buf0_re[i0] = val.x;
        buf0_im[i0] = val.y;
        buf0_re[i1] = val.z;
        buf0_im[i1] = val.w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    stockham_dif_butterflies(buf0_re, buf0_im, buf1_re, buf1_im, idx, N, logN);
    
    threadgroup float *out_re = ((logN & 1u) != 0u) ? buf1_re : buf0_re;
    threadgroup float *out_im = ((logN & 1u) != 0u) ? buf1_im : buf0_im;
    
    if (idx < (N >> 1u)) {
        uint dst_vec = (line << (logN - 1u)) | idx;
        uint i0 = idx << 1u;
        uint i1 = i0 | 1u;
        float4 val;
        val.x = out_re[i0];
        val.y = out_im[i0];
        val.z = out_re[i1];
        val.w = out_im[i1];
        ((device float4*)out_data)[dst_vec] = val;
    }
}

[[max_total_threads_per_threadgroup(1024)]]
kernel void fft3d_y(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) 
{
    uint idx = gid.x;
    uint line = gid.y;
    uint logN = ctz(N);
    
    uint k = line >> logN;
    uint i_coord = line & (N - 1u);
    
    threadgroup float buf0_re[1024];
    threadgroup float buf0_im[1024];
    threadgroup float buf1_re[1024];
    threadgroup float buf1_im[1024];
    
    if (idx < N) {
        uint src = (k << (logN << 1u)) | (idx << logN) | i_coord;
        float2 val = in_data[src];
        buf0_re[idx] = val.x;
        buf0_im[idx] = val.y;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    stockham_dif_butterflies(buf0_re, buf0_im, buf1_re, buf1_im, idx, N, logN);
    
    threadgroup float *out_re = ((logN & 1u) != 0u) ? buf1_re : buf0_re;
    threadgroup float *out_im = ((logN & 1u) != 0u) ? buf1_im : buf0_im;
    
    if (idx < N) {
        uint dst = (k << (logN << 1u)) | (idx << logN) | i_coord;
        out_data[dst] = float2(out_re[idx], out_im[idx]);
    }
}

[[max_total_threads_per_threadgroup(1024)]]
kernel void fft3d_z(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) 
{
    uint idx = gid.x;
    uint line = gid.y;
    uint logN = ctz(N);
    
    threadgroup float buf0_re[1024];
    threadgroup float buf0_im[1024];
    threadgroup float buf1_re[1024];
    threadgroup float buf1_im[1024];
    
    if (idx < N) {
        uint src = (idx << (logN << 1u)) | line;
        float2 val = in_data[src];
        buf0_re[idx] = val.x;
        buf0_im[idx] = val.y;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    stockham_dif_butterflies(buf0_re, buf0_im, buf1_re, buf1_im, idx, N, logN);
    
    threadgroup float *out_re = ((logN & 1u) != 0u) ? buf1_re : buf0_re;
    threadgroup float *out_im = ((logN & 1u) != 0u) ? buf1_im : buf0_im;
    
    if (idx < N) {
        uint dst = (idx << (logN << 1u)) | line;
        out_data[dst] = float2(out_re[idx], out_im[idx]);
    }
}
```

Result of previous attempt:
            32cube: correct, 0.16 ms, 19.3 GB/s (effective, 96 B/cell across 3 passes) (9.6% of 200 GB/s)
            64cube: correct, 0.78 ms, 32.5 GB/s (effective, 96 B/cell across 3 passes) (16.2% of 200 GB/s)
           128cube: correct, 3.16 ms, 63.8 GB/s (effective, 96 B/cell across 3 passes) (31.9% of 200 GB/s)
  score (gmean of fraction): 0.1708

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
- iter  2: compile=OK | correct=True | score=0.2202371411899508
- iter  3: compile=OK | correct=True | score=0.1610986371750679
- iter  4: compile=OK | correct=True | score=0.15556101155007207
- iter  5: compile=OK | correct=True | score=0.1708311951653751

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
