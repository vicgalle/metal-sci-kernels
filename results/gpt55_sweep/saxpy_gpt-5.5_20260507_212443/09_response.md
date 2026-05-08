I’m replacing the per-SIMD in-bounds fast path with a whole-dispatch fast path using `threads_per_grid`. When the padded dispatch size exactly equals `N`—common for power-of-two benchmark sizes—every launched thread is valid, so the hot path avoids the per-element `i < N` guard entirely. For non-exact padded grids, it falls back to the safe guarded scalar path.

```metal
#include <metal_stdlib>
using namespace metal;

kernel void saxpy(device const float * __restrict x [[buffer(0)]],
                  device float       * __restrict y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]],
                  uint gridN [[threads_per_grid]])
{
    const uint n = N;
    const float aa = a;

    // If the dispatch is not padded, every launched thread is in-bounds.
    if (gridN == n) {
        const float xi = x[i];
        const float yi = y[i];
        y[i] = fma(aa, xi, yi);
    } else {
        if (i < n) {
            const float xi = x[i];
            const float yi = y[i];
            y[i] = fma(aa, xi, yi);
        }
    }
}
```