**Optimization:** Add `__restrict__` aliasing hints (matching incumbent) and provide a `max_total_threads_per_threadgroup(1024)` hint to the compiler. The restrict qualifier tells the compiler that x and y don't alias, allowing better scheduling of independent loads. The threadgroup hint helps register allocation. Since this is purely BW-bound and we must do exactly one element per thread, the kernel body itself is already minimal — the wins come from helping the compiler emit the tightest possible load/FMA/store sequence with no aliasing-related reload of y.

**Why better:** The incumbent uses `__restrict__` but no threads-per-threadgroup hint; the previous attempt used the hint but no `__restrict__`. Combining both should give the compiler maximum freedom while constraining occupancy optimally.

```metal
#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(1024)]]
kernel void saxpy(device const float * __restrict__ x [[buffer(0)]],
                  device float       * __restrict__ y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) {
    if (i >= N) return;
    y[i] = fma(a, x[i], y[i]);
}
```