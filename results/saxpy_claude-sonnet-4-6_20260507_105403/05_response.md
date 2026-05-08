Looking at the results, the 1M case (142.6 GB/s, 71.3%) is the bottleneck — likely suffering from latency rather than bandwidth. The incumbent's float4 trick helps large sizes but the branchy logic hurts small sizes. 

Key insight: the previous approach has threads doing redundant work (non-leader threads compute `i & ~3u` and check conditions). For the 1M case, kernel launch overhead and memory latency dominate. I'll use a clean scalar approach with `[[max_total_threads_per_threadgroup(1024)]]` to maximize occupancy, plus straight `fma` — no branching overhead for followers. The float4 approach actually wastes 3/4 of threads in the vector path. Let me instead use `float4` properly by having each thread process exactly one `float4` (4 elements) — but since the spec says one thread per element, I should use the simplest possible kernel with maximum occupancy and let the GPU's memory coalescing hardware do its job.

The real fix: eliminate all branch divergence. A simple, branchless (after the bounds check) `fma` with `[[max_total_threads_per_threadgroup(1024)]]` maximizes occupancy and lets the hardware coalesce naturally. For the 1M case, reducing overhead matters more than vectorization tricks.

```metal
#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(1024)]]
kernel void saxpy(device const float *x [[buffer(0)]],
                  device float       *y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]])
{
    if (i >= N) return;
    y[i] = fma(a, x[i], y[i]);
}
```