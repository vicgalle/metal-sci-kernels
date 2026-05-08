Looking at the results, the small (1M) case is the weakest point (~71-72% efficiency) due to launch overhead and poor occupancy, while the large (64M) case is already near 94%. The key insight: the incumbent's "leader/follower" approach wastes 75% of threads in the vector region doing nothing — those threads still consume GPU resources. Better to keep every thread busy with a simple scalar `fma` using `fma()` and let the GPU's memory coalescing handle vectorization automatically. The real win is to reduce the cost for smaller sizes by using `[[max_total_threads_per_threadgroup(1024)]]` and ensuring the compiler sees a clean, simple loop body it can auto-vectorize at the warp level. I'll go back to the clean scalar approach from iter 9 but add explicit `[[nodiscard]]` prefetch hints and ensure aligned access patterns.

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