I’m applying a SIMD-group broadcast for the uniform scalar arguments `a` and `N`. Since every lane in a SIMD-group uses the same values, broadcasting from lane 0 can avoid redundant constant-buffer traffic and may slightly reduce pressure on the memory/constant path while preserving the one-thread-per-element SAXPY mapping.

```metal
#include <metal_stdlib>
using namespace metal;

kernel void saxpy(device const float * __restrict x [[buffer(0)]],
                  device float       * __restrict y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]])
{
    const uint n = simd_broadcast(N, ushort(0));

    if (i < n) {
        const float aa = simd_broadcast(a, ushort(0));
        const float xi = x[i];
        const float yi = y[i];
        y[i] = fma(aa, xi, yi);
    }
}
```