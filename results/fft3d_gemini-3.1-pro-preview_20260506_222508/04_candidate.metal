#include <metal_stdlib>
using namespace metal;

constant float TWO_PI = 6.28318530717958647692f;

inline uint pad(uint x) {
    return x + (x >> 5u);
}

inline uint bit_reverse(uint x, uint logN) {
    return reverse_bits(x) >> (32u - logN);
}

inline void line_fft_butterflies(threadgroup float2 *buf,
                                 uint i, uint n, uint logN) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 1u; s <= logN; ++s) {
        uint half_m = 1u << (s - 1u);
        
        // Fast exact 2^{-s} using IEEE-754 exponent injection
        float inv_m = as_type<float>((127u - s) << 23u);
        float angle_mult = -TWO_PI * inv_m;
        
        if (i < (n >> 1u)) {
            uint group = i >> (s - 1u);
            uint kk    = i & (half_m - 1u);
            uint base  = group << s;
            
            float angle = angle_mult * float(kk);
            float c, s_;
            s_ = sincos(angle, c);
            float2 w = float2(c, s_);
            
            uint raw_idx0 = base + kk;
            uint raw_idx1 = raw_idx0 + half_m;
            
            uint idx0 = pad(raw_idx0);
            uint idx1 = pad(raw_idx1);
            
            float2 u = buf[idx0];
            float2 v = buf[idx1];
            
            float2 t;
            t.x = fma(-w.y, v.y, w.x * v.x);
            t.y = fma( w.y, v.x, w.x * v.y);
            
            buf[idx0] = u + t;
            buf[idx1] = u - t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

[[max_total_threads_per_threadgroup(1024)]]
kernel void fft3d_x(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint n = N;
    uint logN = ctz(n);
    uint i = gid.x;
    uint line = gid.y;
    
    threadgroup float2 buf[1056];
    
    if (i < n) {
        uint src = (line << logN) | i;
        buf[pad(bit_reverse(i, logN))] = in_data[src];
    }
    
    line_fft_butterflies(buf, i, n, logN);
    
    if (i < n) {
        uint dst = (line << logN) | i;
        out_data[dst] = buf[pad(i)];
    }
}

[[max_total_threads_per_threadgroup(1024)]]
kernel void fft3d_y(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint n = N;
    uint logN = ctz(n);
    uint j = gid.x;
    uint line = gid.y;
    
    uint k = line >> logN;
    uint i = line & (n - 1u);
    
    threadgroup float2 buf[1056];
    
    if (j < n) {
        uint src = (k << (logN << 1u)) | (j << logN) | i;
        buf[pad(bit_reverse(j, logN))] = in_data[src];
    }
    
    line_fft_butterflies(buf, j, n, logN);
    
    if (j < n) {
        uint dst = (k << (logN << 1u)) | (j << logN) | i;
        out_data[dst] = buf[pad(j)];
    }
}

[[max_total_threads_per_threadgroup(1024)]]
kernel void fft3d_z(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint n = N;
    uint logN = ctz(n);
    uint k = gid.x;
    uint line = gid.y;
    
    threadgroup float2 buf[1056];
    
    if (k < n) {
        uint src = (k << (logN << 1u)) | line;
        buf[pad(bit_reverse(k, logN))] = in_data[src];
    }
    
    line_fft_butterflies(buf, k, n, logN);
    
    if (k < n) {
        uint dst = (k << (logN << 1u)) | line;
        out_data[dst] = buf[pad(k)];
    }
}