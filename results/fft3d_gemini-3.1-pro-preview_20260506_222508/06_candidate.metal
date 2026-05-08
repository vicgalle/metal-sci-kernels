#include <metal_stdlib>
using namespace metal;

constant float TWO_PI = 6.28318530717958647692f;

// Executes log2(N) stages of Radix-2 Decimation-In-Time (DIT) butterflies.
// `buf_re` and `buf_im` must already contain the bit-reversed input data.
inline void line_fft_butterflies(threadgroup float *buf_re,
                                 threadgroup float *buf_im,
                                 uint i, uint N, uint logN) {
    for (uint s = 1u; s <= logN; ++s) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint half_m = 1u << (s - 1u);
        uint m      = half_m << 1u;
        
        if (i < (N >> 1u)) {
            uint group = i >> (s - 1u);
            uint kk    = i & (half_m - 1u);
            uint p     = (group << s) + kk;
            uint q     = p + half_m;
            
            float angle = -TWO_PI * float(kk) / float(m);
            float c, s_;
            s_ = sincos(angle, c);
            
            float u_r = buf_re[p];
            float u_i = buf_im[p];
            float v_r = buf_re[q];
            float v_i = buf_im[q];
            
            float t_r = c * v_r - s_ * v_i;
            float t_i = c * v_i + s_ * v_r;
            
            buf_re[p] = u_r + t_r;
            buf_im[p] = u_i + t_i;
            buf_re[q] = u_r - t_r;
            buf_im[q] = u_i - t_i;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

[[max_total_threads_per_threadgroup(1024)]]
kernel void fft3d_x(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint line = gid.y;
    uint logN = ctz(N);
    
    // SoA layout perfectly resolves 2-way bank conflicts on contiguous sequential reads
    threadgroup float buf_re[1024];
    threadgroup float buf_im[1024];

    if (i < N) {
        // Fast exact address decoding without integer div/mod
        uint src = (line << logN) | i;
        uint rev = reverse_bits(i) >> (32u - logN);
        float2 val = in_data[src];
        buf_re[rev] = val.x;
        buf_im[rev] = val.y;
    }
    
    line_fft_butterflies(buf_re, buf_im, i, N, logN);
    
    if (i < N) {
        uint dst = (line << logN) | i;
        out_data[dst] = float2(buf_re[i], buf_im[i]);
    }
}

[[max_total_threads_per_threadgroup(1024)]]
kernel void fft3d_y(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint j = gid.x;
    uint line = gid.y;
    uint logN = ctz(N);
    uint logN2 = logN << 1u;
    
    // Bitwise properties of the line geometry: line = k * N + i
    uint k = line >> logN;
    uint i = line & (N - 1u);
    
    threadgroup float buf_re[1024];
    threadgroup float buf_im[1024];

    if (j < N) {
        uint src = (k << logN2) | (j << logN) | i;
        uint rev = reverse_bits(j) >> (32u - logN);
        float2 val = in_data[src];
        buf_re[rev] = val.x;
        buf_im[rev] = val.y;
    }
    
    line_fft_butterflies(buf_re, buf_im, j, N, logN);
    
    if (j < N) {
        uint dst = (k << logN2) | (j << logN) | i;
        out_data[dst] = float2(buf_re[j], buf_im[j]);
    }
}

[[max_total_threads_per_threadgroup(1024)]]
kernel void fft3d_z(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint k = gid.x;
    uint line = gid.y;
    uint logN = ctz(N);
    uint logN2 = logN << 1u;
    
    threadgroup float buf_re[1024];
    threadgroup float buf_im[1024];

    if (k < N) {
        // Since line = j * N + i, k * N^2 + j * N + i is equivalently just (k * N^2) + line
        uint src = (k << logN2) | line;
        uint rev = reverse_bits(k) >> (32u - logN);
        float2 val = in_data[src];
        buf_re[rev] = val.x;
        buf_im[rev] = val.y;
    }
    
    line_fft_butterflies(buf_re, buf_im, k, N, logN);
    
    if (k < N) {
        uint dst = (k << logN2) | line;
        out_data[dst] = float2(buf_re[k], buf_im[k]);
    }
}