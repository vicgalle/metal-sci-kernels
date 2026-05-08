#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(1024)]]
kernel void saxpy(device const float * __restrict__ x [[buffer(0)]],
                  device float       * __restrict__ y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]],
                  uint tg_id [[threadgroup_position_in_grid]],
                  uint lid [[thread_position_in_threadgroup]],
                  uint tg_size [[threads_per_threadgroup]]) {
    // Base index of this threadgroup
    uint base = tg_id * tg_size;

    // Fast path: entire threadgroup tile fits within N and is float4-aligned.
    // tg_size is 1024 (multiple of 4), base is a multiple of tg_size.
    if (base + tg_size <= N && (base & 3u) == 0u) {
        uint vec_count = tg_size >> 2;          // 256 float4s per tile
        if (lid < vec_count) {
            uint vidx = (base >> 2) + lid;
            device const float4 *xv = reinterpret_cast<device const float4*>(x);
            device float4       *yv = reinterpret_cast<device float4*>(y);
            float4 xv_val = xv[vidx];
            float4 yv_val = yv[vidx];
            yv_val.x = fma(a, xv_val.x, yv_val.x);
            yv_val.y = fma(a, xv_val.y, yv_val.y);
            yv_val.z = fma(a, xv_val.z, yv_val.z);
            yv_val.w = fma(a, xv_val.w, yv_val.w);
            yv[vidx] = yv_val;
        }
        // Threads with lid >= vec_count idle (still launched per spec).
        return;
    }

    // Tail / unaligned path: scalar fallback per element.
    if (i < N) {
        y[i] = fma(a, x[i], y[i]);
    }
}