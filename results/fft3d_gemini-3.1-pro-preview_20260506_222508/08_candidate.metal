#include <metal_stdlib>
using namespace metal;

#define PAD(x) ((x) + ((x) >> 5u))

inline void fft_simd(uint i, uint logN, threadgroup float2* buf, thread float2 &u) {
    // Stage 1
    if (logN >= 1u) {
        float2 v = simd_shuffle_xor(u, 1u);
        u = ((i & 1u) == 0u) ? (u + v) : (v - u);
    }
    
    // Stage 2
    if (logN >= 2u) {
        float2 v = simd_shuffle_xor(u, 2u);
        float2 t;
        if ((i & 2u) == 0u) {
            t = ((i & 1u) == 0u) ? v : float2(v.y, -v.x);
            u = u + t;
        } else {
            t = ((i & 1u) == 0u) ? u : float2(u.y, -u.x);
            u = v - t;
        }
    }
    
    // Stage 3
    if (logN >= 3u) {
        float2 v = simd_shuffle_xor(u, 4u);
        uint kk = i & 3u;
        float angle = -0.7853981633974483f * float(kk); // -2pi / 8
        float c; float s_ = sincos(angle, c);
        float2 t;
        if ((i & 4u) == 0u) {
            t = float2(c * v.x - s_ * v.y, c * v.y + s_ * v.x);
            u = u + t;
        } else {
            t = float2(c * u.x - s_ * u.y, c * u.y + s_ * u.x);
            u = v - t;
        }
    }
    
    // Stage 4
    if (logN >= 4u) {
        float2 v = simd_shuffle_xor(u, 8u);
        uint kk = i & 7u;
        float angle = -0.39269908169872414f * float(kk); // -2pi / 16
        float c; float s_ = sincos(angle, c);
        float2 t;
        if ((i & 8u) == 0u) {
            t = float2(c * v.x - s_ * v.y, c * v.y + s_ * v.x);
            u = u + t;
        } else {
            t = float2(c * u.x - s_ * u.y, c * u.y + s_ * u.x);
            u = v - t;
        }
    }
    
    // Stage 5
    if (logN >= 5u) {
        float2 v = simd_shuffle_xor(u, 16u);
        uint kk = i & 15u;
        float angle = -0.19634954084936207f * float(kk); // -2pi / 32
        float c; float s_ = sincos(angle, c);
        float2 t;
        if ((i & 16u) == 0u) {
            t = float2(c * v.x - s_ * v.y, c * v.y + s_ * v.x);
            u = u + t;
        } else {
            t = float2(c * u.x - s_ * u.y, c * u.y + s_ * u.x);
            u = v - t;
        }
    }
    
    // Stages 6 to logN
    uint toggle_offset = 0;
    for (uint s = 6u; s <= logN; ++s) {
        buf[toggle_offset + PAD(i)] = u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        uint half_m = 1u << (s - 1u);
        uint buddy = i ^ half_m;
        float2 v = buf[toggle_offset + PAD(buddy)];
        
        uint kk = i & (half_m - 1u);
        float angle = -6.28318530717958647692f * float(kk) / float(half_m * 2u);
        float c; float s_ = sincos(angle, c);
        
        float2 t;
        if ((i & half_m) == 0u) {
            t = float2(c * v.x - s_ * v.y, c * v.y + s_ * v.x);
            u = u + t;
        } else {
            t = float2(c * u.x - s_ * u.y, c * u.y + s_ * u.x);
            u = v - t;
        }
        
        // Ping-pong between 0 and 1056 offset in the 2112-element buffer
        toggle_offset = 1056u - toggle_offset;
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
    
    // 2x padding allocation for ping-pong double-buffering
    threadgroup float2 buf[2112];
    
    uint src = (line << logN) | i;
    uint rev = reverse_bits(i) >> (32u - logN);
    
    // Initial global load perfectly mapped to avoid 32-way bank conflicts
    buf[PAD(rev)] = in_data[src];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float2 u = buf[PAD(i)];
    fft_simd(i, logN, buf, u);
    out_data[src] = u;
}

[[max_total_threads_per_threadgroup(1024)]]
kernel void fft3d_y(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint j = gid.x;
    uint line = gid.y;
    uint logN = ctz(N);
    
    threadgroup float2 buf[2112];
    
    uint k = line >> logN;
    uint i_idx = line & (N - 1u);
    uint src = (k << (2u * logN)) | (j << logN) | i_idx;
    
    uint rev = reverse_bits(j) >> (32u - logN);
    buf[PAD(rev)] = in_data[src];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float2 u = buf[PAD(j)];
    fft_simd(j, logN, buf, u);
    out_data[src] = u;
}

[[max_total_threads_per_threadgroup(1024)]]
kernel void fft3d_z(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint k = gid.x;
    uint line = gid.y;
    uint logN = ctz(N);
    
    threadgroup float2 buf[2112];
    
    uint src = (k << (2u * logN)) | line;
    
    uint rev = reverse_bits(k) >> (32u - logN);
    buf[PAD(rev)] = in_data[src];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float2 u = buf[PAD(k)];
    fft_simd(k, logN, buf, u);
    out_data[src] = u;
}