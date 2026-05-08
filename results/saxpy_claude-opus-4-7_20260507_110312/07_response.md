**Optimization:** Add `[[max_total_threads_per_threadgroup(1024)]]` hint to give the compiler more flexibility for occupancy tuning, and use a branchless approach via `min(i, N-1)` style masking — actually, the cleanest win for a pure memory-bound kernel is to ensure the compiler emits the tightest possible load/fma/store sequence. I'll keep the guard but mark it `unlikely` via structuring, and add the threadgroup hint. The previous best (0.834) used a simple guard + fma; I'll preserve that core but add compiler hints to help scheduling and try to coalesce better.

**Why:** SAXPY is purely BW-bound at large N. The 1M case underperforms (72%) due to launch/occupancy overhead — boosting threadgroup size hint helps the scheduler pack more threads. For 16M/64M we're already near peak; the hint shouldn't hurt and may give the driver more flexibility.

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
    const float xi = x[i];
    const float yi = y[i];
    y[i] = fma(a, xi, yi);
}
```