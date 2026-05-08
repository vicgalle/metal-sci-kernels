**Optimization:** Use `fma()` for fused multiply-add and add `[[max_total_threads_per_threadgroup]]` hint. Since each thread must handle exactly one element (grid is fixed by host), the main lever is using `fma` for a single FMA instruction and ensuring the compiler generates optimal load/store sequences. The `fma` intrinsic maps directly to a hardware FMA, slightly reducing instruction count vs. separate mul+add, though for a memory-bound kernel the dominant cost is bandwidth. The threadgroup hint helps the compiler allocate registers tightly for max occupancy.

**Why better:** Saxpy is bandwidth-bound; the seed is already at 90% of peak at 64M. Remaining headroom comes from reducing per-thread instruction overhead (better FMA codegen) and improving small-size launch efficiency via the threads-per-threadgroup hint, which lets the compiler optimize for higher occupancy on Apple GPUs.

```metal
#include <metal_stdlib>
using namespace metal;

kernel void saxpy(device const float *x [[buffer(0)]],
                  device float       *y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) [[max_total_threads_per_threadgroup(1024)]] {
    if (i >= N) return;
    float xi = x[i];
    float yi = y[i];
    y[i] = fma(a, xi, yi);
}
```