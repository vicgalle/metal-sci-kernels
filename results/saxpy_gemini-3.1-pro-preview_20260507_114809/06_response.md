To optimize the SAXPY kernel under the strict 1D 1-element-per-thread dispatch constraint, I apply two main changes. First, I replace the predicated `if (i < N)` block from the previous attempt with an early-exit `if (i >= N) return;`. This allows entirely out-of-bounds SIMD groups to immediately terminate and avoids predication overhead on the main execution path. Second, I explicitly use the `fma()` intrinsic instead of `a * x[i] + y[i]`. While compilers often contract this automatically, explicitly calling `fma()` guarantees the emission of a single hardware fused multiply-add instruction, minimizing latency. The signature exactly matches the specification without introducing potentially problematic modifiers.

```metal
#include <metal_stdlib>
using namespace metal;

kernel void saxpy(device const float *x [[buffer(0)]],
                  device float       *y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) {
    if (i >= N) return;
    y[i] = fma(a, x[i], y[i]);
}
```