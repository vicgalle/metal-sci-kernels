**Optimization**: The naive incumbent is already very fast because it's purely memory-bound and the L1/L2 cache on Apple Silicon handles the 5-point stencil reuse well. My previous tiled attempt was slower due to barrier overhead and complex halo loads. Instead of threadgroup tiling, I'll try a vectorized approach: each thread still computes one cell (per spec), but I'll use `float4` vector loads where possible... wait, that violates "one thread per cell."

**Better approach**: Stick close to the naive kernel but (1) hint threadgroup size for better occupancy, (2) precompute row offsets to reduce index arithmetic, (3) use a single branch for boundaries with `select`, and (4) read center via the same index reused for write to help the compiler coalesce. Keeping the kernel simple lets the hardware caches do the reuse work, which beat my explicit tiling.

```metal
#include <metal_stdlib>
using namespace metal;

kernel void heat_step(device const float *u_in  [[buffer(0)]],
                      device       float *u_out [[buffer(1)]],
                      constant uint      &NX    [[buffer(2)]],
                      constant uint      &NY    [[buffer(3)]],
                      constant float     &alpha [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint j = gid.y;
    if (i >= NX || j >= NY) return;

    uint row    = j * NX;
    uint idx    = row + i;

    float c = u_in[idx];

    bool boundary = (i == 0) | (j == 0) | (i == NX - 1) | (j == NY - 1);
    if (boundary) {
        u_out[idx] = c;
        return;
    }

    float l = u_in[idx - 1];
    float r = u_in[idx + 1];
    float d = u_in[idx - NX];
    float uu = u_in[idx + NX];

    u_out[idx] = c + alpha * ((l + r) + (d + uu) - 4.0f * c);
}
```