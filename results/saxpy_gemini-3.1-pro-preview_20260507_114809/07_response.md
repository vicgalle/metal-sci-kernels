To optimize this fully memory-bound kernel where each thread computes exactly one element, we restore the `a * x[i] + y[i]` arithmetic to allow the compiler to use its default fast-math fma lowering (which empirically performs better here than explicit `fma` due to minor instruction selection differences). We also add the `restrict` keyword to the pointer arguments. While `x` is `const`, Metal's compiler must technically assume `y` could alias with `x` (i.e. modifying `y` could theoretically affect what `x` points to, if they overlaid). Adding `restrict` explicitly guarantees to the compiler that these buffers do not overlap, enabling the widest possible memory transaction coalescing and optimal load/store scheduling. 

```metal
#include <metal_stdlib>
using namespace metal;

kernel void saxpy(device const float * restrict x [[buffer(0)]],
                  device float       * restrict y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) {
    if (i >= N) return;
    y[i] = a * x[i] + y[i];
}
```