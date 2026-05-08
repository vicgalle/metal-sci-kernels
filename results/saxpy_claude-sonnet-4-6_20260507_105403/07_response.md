Looking at the history, the incumbent (iter 2) uses the leader-thread float4 approach and scores 0.8308. My previous attempt (iter 6) tried a different float4 remapping but scored worse at 0.7046. I need to beat the incumbent.

The key insight: SAXPY is purely memory-bandwidth bound. The incumbent already uses float4 vectorization. To go further, I should try `float4` loads with explicit `[[max_total_threads_per_threadgroup]]` hint and ensure the compiler sees the access pattern cleanly. More importantly, I can try reading with `metal::simdgroup` cooperation or use `float4` with non-temporal store hints.

The main opportunity: the small (1M) case is only at 71.3% while the large (64M) case hits 93.4%. For small sizes, latency/occupancy dominates. I'll keep the incumbent's leader-thread float4 approach but add `[[max_total_threads_per_threadgroup(1024)]]` to improve occupancy, and try using `as_type<float4>` to encourage the compiler to generate optimal load/store instructions.

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

    if ((i & 3u) == 0u) {
        uint remaining = N - i;
        if (remaining >= 4u) {
            device const float4 *xv = reinterpret_cast<device const float4*>(x + i);
            device float4       *yv = reinterpret_cast<device float4*>(y + i);
            float4 xval = *xv;
            float4 yval = *yv;
            *yv = fma(a, xval, yval);
            return;
        }
    } else {
        uint leader = i & ~3u;
        if (leader < N && (N - leader) >= 4u) {
            return;
        }
    }

    y[i] = fma(a, x[i], y[i]);
}
```