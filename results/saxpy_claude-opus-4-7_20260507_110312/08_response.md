**Optimization:** Since the host dispatches exactly one thread per element and we cannot coarsen at the thread level, the only levers are: (1) reducing per-thread instruction overhead, (2) ensuring the compiler emits the tightest possible load-FMA-store sequence, and (3) avoiding any branch when possible. The incumbent already does this well. I'll try removing `__restrict__` (which sometimes hurts vs helps depending on compiler heuristics) and use a `select`-free guard via early return, while adding a `max_total_threads_per_threadgroup(1024)` hint so the compiler can pick the widest threadgroup and minimize launch overhead per element. This matches the BW-bound nature: at 64M we're already at 92.6%, so the goal is mainly reducing fixed overhead at small N (1M).

```metal
#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(1024)]]
kernel void saxpy(device const float *x [[buffer(0)]],
                  device float       *y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) {
    if (i >= N) return;
    y[i] = fma(a, x[i], y[i]);
}
```