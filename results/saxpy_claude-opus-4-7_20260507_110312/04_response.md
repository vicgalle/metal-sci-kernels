The keybindings-help skill is not relevant to this MSL kernel optimization task.

**Optimization:** Since each thread must process exactly one element (host dispatches one thread per element), I can't vectorize across threads. The naive kernel is already simple, but I can hint the compiler with `[[max_total_threads_per_threadgroup]]` set to match Apple's preferred SIMD-multiple sizing, and use a non-temporal-style straightforward FMA. The key insight: avoid the early-return branch divergence pattern by using a min-clamped approach that still respects the bound. Actually, the cleanest path is to keep the guard but ensure the compiler emits a single FMA instruction. I'll use `fma()` explicitly and add the threadgroup hint, which the incumbent lacks. The incumbent (no hint) outperformed my previous (1024 hint), so I'll try a hint of 256 which is often the sweet spot for memory-bound kernels on Apple GPUs, allowing more threadgroups in flight to hide memory latency.

```metal
#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(256)]]
kernel void saxpy(device const float *x [[buffer(0)]],
                  device float       *y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) {
    if (i >= N) return;
    float xi = x[i];
    float yi = y[i];
    y[i] = fma(a, xi, yi);
}
```