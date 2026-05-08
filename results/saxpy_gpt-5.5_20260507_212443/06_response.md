Applies a uniform fast path using `threads_per_grid`: when the padded dispatch size equals `N` (true for the benchmark power-of-two sizes), the kernel skips the per-thread bounds check entirely. A guarded fallback preserves correctness for padded tails. This should reduce a small amount of ALU/control overhead without changing the bandwidth-bound memory access pattern.

```metal
#include <metal_stdlib>
using namespace metal;

kernel void saxpy(device const float * __restrict x [[buffer(0)]],
                  device float       * __restrict y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]],
                  uint grid_size [[threads_per_grid]])
{
    const uint n = N;
    const float aa = a;

    if (grid_size != n) {
        if (i >= n) {
            return;
        }
    }

    y[i] = fma(aa, x[i], y[i]);
}
```