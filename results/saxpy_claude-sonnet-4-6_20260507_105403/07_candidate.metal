#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(1024)]]
kernel void saxpy(device const float *x [[buffer(0)]],
                  device float       *y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]])
{
    if (i >= N) return;

    if ((i & 3u) == 0u) {
        uint remaining = N - i;
        if (remaining >= 4u) {
            device const float4 *xv = reinterpret_cast<device const float4*>(x + i);
            device float4       *yv = reinterpret_cast<device float4*>(y + i);
            float4 xval = *xv;
            float4 yval = *yv;
            *yv = fma(a, xval, yval);
            return;
        }
    } else {
        uint leader = i & ~3u;
        if (leader < N && (N - leader) >= 4u) {
            return;
        }
    }

    y[i] = fma(a, x[i], y[i]);
}