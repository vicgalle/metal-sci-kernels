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