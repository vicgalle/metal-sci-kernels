I’m keeping the SAXPY path minimal and bandwidth-oriented, with `restrict` to preserve no-alias assumptions and a `max_total_threads_per_threadgroup(256)` hint for the compiler. This should retain the incumbent’s optimal coalesced scalar memory pattern while potentially improving scheduling/occupancy assumptions without adding extra per-thread work.

```metal
#include <metal_stdlib>
using namespace metal;

kernel void saxpy(device const float * __restrict x [[buffer(0)]],
                  device float       * __restrict y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]])
                  [[max_total_threads_per_threadgroup(256)]]
{
    if (i < N) {
        y[i] = fma(a, x[i], y[i]);
    }
}
```