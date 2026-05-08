#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(256)]]
kernel void saxpy(device const float * __restrict x [[buffer(0)]],
                  device float       * __restrict y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]])
{
    const uint n = N;
    if (i >= n) {
        return;
    }

    device float *yp = y + i;
    const float xi = x[i];
    const float yi = *yp;
    const float aa = a;

    *yp = fma(aa, xi, yi);
}