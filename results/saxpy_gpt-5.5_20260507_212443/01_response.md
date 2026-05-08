Use a minimal bandwidth-oriented scalar kernel: keep one thread per element, add `__restrict` to help alias analysis, cache scalar constants in registers, and use explicit `fma` to encourage a single fused multiply-add. This preserves the optimal coalesced streaming access pattern while slightly reducing compiler ambiguity versus the seed.

```metal
#include <metal_stdlib>
using namespace metal;

kernel void saxpy(device const float * __restrict x [[buffer(0)]],
                  device float       * __restrict y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) {
    const uint n = N;
    const float aa = a;

    if (i < n) {
        const float xi = x[i];
        const float yi = y[i];
        y[i] = fma(aa, xi, yi);
    }
}
```