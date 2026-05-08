#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(256)]]
kernel void saxpy(device const float *x [[buffer(0)]],
                  device float       *y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) {
    if (i >= N) return;
    float xi = x[i];
    float yi = y[i];
    y[i] = fma(a, xi, yi);
}