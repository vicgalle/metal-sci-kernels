To push beyond the incumbent, this kernel applies four major optimizations that fundamentally reduce memory traffic and improve ALU utilization:

1. **Warp-level SIMD Butterflies**: The first 5 stages of the FFT (`m=2,4,8,16,32`) only mix elements within a 32-wide SIMD group. By keeping exactly 1 element per thread in registers and communicating via `simd_shuffle_xor`, we completely eliminate shared memory reads/writes and barriers for the first 5 stages.
2. **100% Thread Activity**: The incumbent masks out half the threads (`i < N/2`), whereas this symmetric decimation-in-time formulation uses all `N` threads simultaneously. Each thread computes its own single output per stage.
3. **Perfect Bank-Conflict Padding**: The bit-reversed scatter from global into shared memory normally causes a severe 32-way bank conflict because bit-reversed indices are multiples of 32. Using a mathematically perfect padding mapping `PAD(x) = x + (x >> 5)`, we map this 32-way conflict perfectly down to an optimal 2-way conflict without disrupting contiguous reads later.
4. **Fast Indexing & `sincospi`**: Global memory indices are computed via bitwise shifts, and twiddles are computed using `sincospi`, avoiding the `TWO_PI` multiplication.

```metal
#include <metal_stdlib>
using namespace metal;

#define PAD(x) ((x) + ((x) >> 5u))

inline void fft_simd(uint i, uint logN, threadgroup float2 *buf, thread float2 &u) {
    // Stage 1
    float2 v1 = simd_shuffle_xor(u, 1u);
    u = ((i & 1u) == 0u) ? (u + v1) : (v1 - u);
    
    // Stage 2
    if (logN >= 2u) {
        float2 v2 = simd_shuffle_xor(u, 2u);
        float2 t2;
        if ((i & 2u) == 0u) {
            t2 = ((i & 1u) == 0u) ? v2 : float2(v2.y, -v2.x);
            u = u + t2;
        } else {
            t2 = ((i & 1u) == 0u) ? u : float2(u.y, -u.x);
            u = v2 - t2;
        }
    }
    
    // Stages 3 to 5 (SIMD shuffle, zero shared memory)
    uint max_simd = min(logN, 5u);
    for (uint s = 3u; s <= max_simd; ++s) {
        uint half_m = 1u << (s - 1u);
        float2 v = simd_shuffle_xor(u, half_m);
        uint kk = i & (half_m - 1u);
        
        float angle_pi = -float(kk) / float(half_m);
        float c, s_;
        s_ = sincospi(angle_pi, c);
        
        float2 t;
        if ((i & half_m) == 0u) {
            t = float2(c * v.x - s_ * v.y, c * v.y + s_ * v.x);
            u = u + t;
        } else {
            t = float2(c * u.x - s_ * u.y, c * u.y + s_ * u.x);
            u = v - t;
        }
    }
    
    // Stages 6 to logN (Threadgroup memory)
    for (uint s = 6u; s <= logN; ++s) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf[PAD(i)] = u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        uint half_m = 1u << (s - 1u);
        uint buddy = i ^ half_m;
        float2 v = buf[PAD(buddy)];
        
        uint kk = i & (half_m - 1u);
        float angle_pi = -float(kk) / float(half_m);
        float c, s_;
        s_ = sincospi(angle_pi, c);
        
        float2 t;
        if ((i & half_m) == 0u) {
            t = float2(c * v.x - s_ * v.y, c * v.y + s_ * v.x);
            u = u + t;
        } else {
            t = float2(c * u.x - s_ * u.y, c * u.y + s_ * u.x);
            u = v - t;
        }
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
    
    threadgroup float2 buf[1056];
    
    uint src = (line << logN) | i;
    uint rev = reverse_bits(i) >> (32u - logN);
    
    // Scatter load with optimal pad mapping to eliminate 32-way bank conflicts
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
    
    threadgroup float2 buf[1056];
    
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
    
    threadgroup float2 buf[1056];
    
    uint src = (k << (2u * logN)) | line;
    
    uint rev = reverse_bits(k) >> (32u - logN);
    buf[PAD(rev)] = in_data[src];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float2 u = buf[PAD(k)];
    fft_simd(k, logN, buf, u);
    out_data[src] = u;
}
```