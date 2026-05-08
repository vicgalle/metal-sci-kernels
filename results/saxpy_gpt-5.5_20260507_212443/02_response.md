I’ll keep the one-thread-per-element mapping but make the kernel as compiler-friendly as possible: explicit restrict/no-aliasing, a fast early bounds exit, and a max threadgroup hint that does not constrain typical dispatches. For this bandwidth-bound SAXPY, the main goal is to minimize instruction overhead around the three global memory operations while preserving deterministic FMA semantics.```metal
#include <metal_stdlib>
using namespace metal;

kernel void saxpy(device const float * __restrict x [[buffer(0)]],
                  device float       * __restrict y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]])
                  [[max_total_threads_per_threadgroup(1024)]]
{
    if (i >= N) {
        return;
    }

    y[i] = fma(a, x[i], y[i]);
}
```