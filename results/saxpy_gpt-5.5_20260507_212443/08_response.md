Avoiding the prior constant `simd_broadcast` shuffles, this version uses a SIMD-width bounds fast path: full 32-lane blocks execute the SAXPY body directly, while only the final partial SIMDgroup falls back to the scalar guard. This should reduce tail-predication overhead without changing memory traffic or per-thread ownership.

```metal
#include <metal_stdlib>
using namespace metal;

kernel void saxpy(device const float * __restrict x [[buffer(0)]],
                  device float       * __restrict y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]])
{
    const uint n = N;
    const float aa = a;

    // Fast path for SIMD-width blocks that are entirely in bounds.
    if ((i | 31u) < n) {
        const float xi = x[i];
        const float yi = y[i];
        y[i] = fma(aa, xi, yi);
        return;
    }

    // Tail / padded-grid path.
    if (i < n) {
        const float xi = x[i];
        const float yi = y[i];
        y[i] = fma(aa, xi, yi);
    }
}
```