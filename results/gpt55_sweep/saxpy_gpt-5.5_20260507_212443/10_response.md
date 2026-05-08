Applying a SIMD-group cooperative load pattern: one lane per 4 contiguous elements issues 16-byte vector loads for both `x` and `y`, then broadcasts the values to its neighboring lanes. Each thread still computes and stores exactly its own `y[i]`, while global load lane pressure is reduced and the scalar tail remains guarded.  
```metal
#include <metal_stdlib>
using namespace metal;

kernel void saxpy(device const float * __restrict x [[buffer(0)]],
                  device float       * __restrict y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]])
{
    const uint n  = N;
    const uint n4 = n & ~3u;
    const float aa = a;

    const uint q    = i & 3u;
    const uint lane = i & 31u;
    const ushort src_lane = ushort(lane & ~3u);

    float4 xv = float4(0.0f);
    float4 yv = float4(0.0f);

    if ((q == 0u) && (i < n4)) {
        const uint vi = i >> 2;
        device const float4 *x4 = reinterpret_cast<device const float4 *>(x);
        device const float4 *y4 = reinterpret_cast<device const float4 *>(y);
        xv = x4[vi];
        yv = y4[vi];
    }

    const float4 xb = simd_broadcast(xv, src_lane);
    const float4 yb = simd_broadcast(yv, src_lane);

    if (i < n4) {
        y[i] = fma(aa, xb[q], yb[q]);
    } else if (i < n) {
        y[i] = fma(aa, x[i], y[i]);
    }
}
```