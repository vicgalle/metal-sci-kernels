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