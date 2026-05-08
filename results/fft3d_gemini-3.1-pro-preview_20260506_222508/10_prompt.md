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

#define PAD(x) ((x) + ((x) >> 5u))

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline void butterfly_simd(uint i, uint logN, thread float2 &u) {
    if (logN >= 1u) {
        float2 v = simd_shuffle_xor(u, 1u);
        u = ((i & 1u) == 0u) ? (u + v) : (v - u);
    }
    if (logN >= 2u) {
        float2 v = simd_shuffle_xor(u, 2u);
        float2 val = ((i & 2u) == 0u) ? v : u;
        float2 t = ((i & 1u) == 0u) ? val : float2(val.y, -val.x);
        u = ((i & 2u) == 0u) ? (u + t) : (v - t);
    }
    if (logN >= 3u) {
        float2 v = simd_shuffle_xor(u, 4u);
        uint kk = i & 3u;
        float angle = -TWO_PI * float(kk) / 8.0f;
        float c, s_; 
        s_ = sincos(angle, c);
        float2 w = float2(c, s_);
        u = ((i & 4u) == 0u) ? (u + cmul(w, v)) : (v - cmul(w, u));
    }
    if (logN >= 4u) {
        float2 v = simd_shuffle_xor(u, 8u);
        uint kk = i & 7u;
        float angle = -TWO_PI * float(kk) / 16.0f;
        float c, s_; 
        s_ = sincos(angle, c);
        float2 w = float2(c, s_);
        u = ((i & 8u) == 0u) ? (u + cmul(w, v)) : (v - cmul(w, u));
    }
    if (logN >= 5u) {
        float2 v = simd_shuffle_xor(u, 16u);
        uint kk = i & 15u;
        float angle = -TWO_PI * float(kk) / 32.0f;
        float c, s_; 
        s_ = sincos(angle, c);
        float2 w = float2(c, s_);
        u = ((i & 16u) == 0u) ? (u + cmul(w, v)) : (v - cmul(w, u));
    }
}

inline void perform_fft(uint i, uint N, uint logN, threadgroup float2* buf, thread float2& u) {
    butterfly_simd(i, logN, u);
    
    if (logN > 5u) {
        buf[PAD(i)] = u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for (uint s = 6u; s <= logN; ++s) {
            uint m = 1u << s;
            uint half_m = m >> 1u;
            if (i < (N >> 1u)) {
                uint group = i >> (s - 1u);
                uint kk = i & (half_m - 1u);
                uint base = group << s;
                
                float angle = -TWO_PI * float(kk) / float(m);
                float c, s_;
                s_ = sincos(angle, c);
                float2 w = float2(c, s_);
                
                uint idx0 = PAD(base + kk);
                uint idx1 = PAD(base + half_m + kk);
                
                float2 u_val = buf[idx0];
                float2 v_val = buf[idx1];
                float2 t = cmul(w, v_val);
                
                buf[idx0] = u_val + t;
                buf[idx1] = u_val - t;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        u = buf[PAD(i)];
    }
}

[[max_total_threads_per_threadgroup(1024)]] kernel void fft3d_x(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint line = gid.y;
    uint logN = ctz(N);
    
    uint k = line / N;
    uint j = line - k * N;
    uint src = (k * N + j) * N + i;
    
    threadgroup float2 buf[1056];
    
    uint rev = reverse_bits(i) >> (32u - logN);
    buf[PAD(rev)] = in_data[src];
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float2 u = buf[PAD(i)];
    perform_fft(i, N, logN, buf, u);
    
    out_data[src] = u;
}

[[max_total_threads_per_threadgroup(1024)]] kernel void fft3d_y(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint j = gid.x;
    uint line = gid.y;
    uint logN = ctz(N);
    
    uint k = line / N;
    uint i = line - k * N;
    uint src = (k * N + j) * N + i;
    
    threadgroup float2 buf[1056];
    
    uint rev = reverse_bits(j) >> (32u - logN);
    buf[PAD(rev)] = in_data[src];
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float2 u = buf[PAD(j)];
    perform_fft(j, N, logN, buf, u);
    
    out_data[src] = u;
}

[[max_total_threads_per_threadgroup(1024)]] kernel void fft3d_z(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint k = gid.x;
    uint line = gid.y;
    uint logN = ctz(N);
    
    uint j = line / N;
    uint i = line - j * N;
    uint src = (k * N + j) * N + i;
    
    threadgroup float2 buf[1056];
    
    uint rev = reverse_bits(k) >> (32u - logN);
    buf[PAD(rev)] = in_data[src];
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float2 u = buf[PAD(k)];
    perform_fft(k, N, logN, buf, u);
    
    out_data[src] = u;
}
```

Result of previous attempt:
            32cube: correct, 0.10 ms, 30.3 GB/s (effective, 96 B/cell across 3 passes) (15.1% of 200 GB/s)
            64cube: correct, 0.52 ms, 47.9 GB/s (effective, 96 B/cell across 3 passes) (24.0% of 200 GB/s)
           128cube: correct, 1.70 ms, 118.6 GB/s (effective, 96 B/cell across 3 passes) (59.3% of 200 GB/s)
  score (gmean of fraction): 0.2781

## History

- iter  2: compile=OK | correct=True | score=0.2202371411899508
- iter  3: compile=OK | correct=True | score=0.1610986371750679
- iter  4: compile=OK | correct=True | score=0.15556101155007207
- iter  5: compile=OK | correct=True | score=0.1708311951653751
- iter  6: compile=OK | correct=True | score=0.1834333851880671
- iter  7: compile=FAIL | correct=False | score=N/A
- iter  8: compile=OK | correct=True | score=0.18195666934753865
- iter  9: compile=OK | correct=True | score=0.27811385545578327

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
