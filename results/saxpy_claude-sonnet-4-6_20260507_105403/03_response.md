Looking at the results, the previous attempt achieved 93.4% at 64M but only 71.3% at 1M. The vectorized approach with leader/follower threads is hurting small sizes because 3/4 threads are doing extra work (checking conditions) before returning. The real issue is that simple scalar FMA should saturate bandwidth — let's try a clean, simple scalar kernel with `fma` and let the GPU's memory subsystem handle coalescing naturally. Apple Silicon's memory controller is excellent at coalescing adjacent scalar accesses from a warp. The overhead of the leader/follower branching logic is hurting performance, especially at small sizes.

The key insight: for a memory-bound SAXPY, the bottleneck is DRAM bandwidth, not instruction throughput. A clean scalar kernel with `fma` should perform at least as well, without the branch divergence overhead.

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