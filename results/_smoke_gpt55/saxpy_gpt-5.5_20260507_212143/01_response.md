Applying a conservative optimization: use an explicit fused multiply-add and a threadgroup-size hint. This keeps the one-thread-per-element contract and the same memory traffic, while encouraging the compiler to emit the optimal single FMA datapath and schedule for typical Apple GPU threadgroup widths.  

```metal
#include <metal_stdlib>
using namespace metal;

kernel void saxpy(device const float *x [[buffer(0)]],
                  device float       *y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]])
                  [[max_total_threads_per_threadgroup(256)]] {
    if (i >= N) return;

    const float xi = x[i];
    const float yi = y[i];
    y[i] = fma(a, xi, yi);
}
```