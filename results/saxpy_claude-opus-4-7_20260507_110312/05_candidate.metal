#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(1024)]]
kernel void saxpy(device const float *x [[buffer(0)]],
                  device float       *y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) {
    // Clamp index so loads are always in-bounds and the entire SIMD
    // group issues one coalesced 128-byte transaction without any
    // divergent control flow on the hot path.
    uint idx = (i < N) ? i : (N - 1);
    float xi = x[idx];
    float yi = y[idx];
    float r  = fma(a, xi, yi);
    if (i < N) {
        y[i] = r;
    }
}