1. We increase thread utilization in the shared-memory Cooley-Tukey stages from 50% to 100% by having every thread compute its respective butterfly output.
2. We halve the number of threadgroup barriers in the shared-memory stages by ping-ponging between two halves of a slightly larger threadgroup buffer (`2048` size).
3. We completely eliminate the shared-memory bit-reversal phase for the Y and Z axes, loading directly via a bit-reversed index. Because Y and Z accesses have large memory strides (`N` and `N*N`), they inherently cannot be coalesced within the threadgroup. Loading directly saves shared memory traffic and barriers without worsening cache behavior. The X axis retains the shared memory bit-reversal to preserve its perfect memory coalescing.

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
        float2 val = ((i & 4u) == 0u) ? v : u;
        uint kk = i & 3u;
        float angle = -TWO_PI * float(kk) * 0.125f;
        float c, s_; s_ = sincos(angle, c);
        float2 w = float2(c, s_);
        float2 t = cmul(w, val);
        u = ((i & 4u) == 0u) ? (u + t) : (v - t);
    }
    if (logN >= 4u) {
        float2 v = simd_shuffle_xor(u, 8u);
        float2 val = ((i & 8u) == 0u) ? v : u;
        uint kk = i & 7u;
        float angle = -TWO_PI * float(kk) * 0.0625f;
        float c, s_; s_ = sincos(angle, c);
        float2 w = float2(c, s_);
        float2 t = cmul(w, val);
        u = ((i & 8u) == 0u) ? (u + t) : (v - t);
    }
    if (logN >= 5u) {
        float2 v = simd_shuffle_xor(u, 16u);
        float2 val = ((i & 16u) == 0u) ? v : u;
        uint kk = i & 15u;
        float angle = -TWO_PI * float(kk) * 0.03125f;
        float c, s_; s_ = sincos(angle, c);
        float2 w = float2(c, s_);
        float2 t = cmul(w, val);
        u = ((i & 16u) == 0u) ? (u + t) : (v - t);
    }
}

inline void perform_fft(uint i, uint logN, threadgroup float2* buf, thread float2& u) {
    butterfly_simd(i, logN, u);
    
    uint buf_idx = 0u;
    for (uint s = 6u; s <= logN; ++s) {
        buf[buf_idx * 1024u + i] = u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        uint half_m = 1u << (s - 1u);
        float2 v = buf[buf_idx * 1024u + (i ^ half_m)];
        
        uint kk = i & (half_m - 1u);
        float angle = -TWO_PI * float(kk) / float(half_m << 1u);
        float c, s_;
        s_ = sincos(angle, c);
        float2 w = float2(c, s_);
        
        u = ((i & half_m) == 0u) ? (u + cmul(w, v)) : (v - cmul(w, u));
        
        buf_idx = 1u - buf_idx;
    }
}

[[max_total_threads_per_threadgroup(1024)]]
kernel void fft3d_x(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint line = gid.y;
    uint logN = ctz(N);
    
    uint k = line / N;
    uint j = line - k * N;
    
    threadgroup float2 buf[2048];
    
    uint src_linear = (k * N + j) * N + i;
    buf[PAD(i)] = in_data[src_linear];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    uint rev = reverse_bits(i) >> (32u - logN);
    float2 u = buf[PAD(rev)];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    perform_fft(i, logN, buf, u);
    
    out_data[src_linear] = u;
}

[[max_total_threads_per_threadgroup(1024)]]
kernel void fft3d_y(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint j = gid.x;
    uint line = gid.y;
    uint logN = ctz(N);
    
    uint k = line / N;
    uint i = line - k * N;
    
    uint rev = reverse_bits(j) >> (32u - logN);
    uint src_in = (k * N + rev) * N + i;
    float2 u = in_data[src_in];
    
    threadgroup float2 buf[2048];
    
    perform_fft(j, logN, buf, u);
    
    uint src_out = (k * N + j) * N + i;
    out_data[src_out] = u;
}

[[max_total_threads_per_threadgroup(1024)]]
kernel void fft3d_z(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint k = gid.x;
    uint line = gid.y;
    uint logN = ctz(N);
    
    uint j = line / N;
    uint i = line - j * N;
    
    uint rev = reverse_bits(k) >> (32u - logN);
    uint src_in = (rev * N + j) * N + i;
    float2 u = in_data[src_in];
    
    threadgroup float2 buf[2048];
    
    perform_fft(k, logN, buf, u);
    
    uint src_out = (k * N + j) * N + i;
    out_data[src_out] = u;
}
```