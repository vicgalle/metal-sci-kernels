Looking at the previous attempt, the float4 "leader/follower" trick is actually wasteful — follower threads do no useful work but still occupy GPU execution slots. The better approach is a clean scalar kernel with `fma`, letting the GPU's memory subsystem handle coalesced 128-bit cache-line reads naturally. The real bottleneck is memory bandwidth, so we should minimize instruction overhead and let the hardware prefetcher work. I'll also try using `[[thread_position_in_grid]]` with a simple guard and trust the compiler to vectorize aligned accesses.

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