#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(1024)]]
kernel void saxpy(device const float * __restrict__ x [[buffer(0)]],
                  device float       * __restrict__ y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) {
    if (i >= N) return;
    y[i] = fma(a, x[i], y[i]);
}