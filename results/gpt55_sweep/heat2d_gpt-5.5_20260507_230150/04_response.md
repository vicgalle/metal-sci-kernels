The previous shuffle-heavy version added control-flow and shuffle overhead that outweighed cache reuse. This version keeps the fast scalar stencil, hoists common address computation, uses a single center load for boundary/interior handling, and keeps loads scheduled simply so Apple’s cache/coalescer can do the work efficiently.

```metal
#include <metal_stdlib>
using namespace metal;

kernel void heat_step(device const float *u_in  [[buffer(0)]],
                      device       float *u_out [[buffer(1)]],
                      constant uint      &NX    [[buffer(2)]],
                      constant uint      &NY    [[buffer(3)]],
                      constant float     &alpha [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]]) {
    const uint i  = gid.x;
    const uint j  = gid.y;
    const uint nx = NX;
    const uint ny = NY;

    if (i >= nx || j >= ny) return;

    const uint idx = j * nx + i;

    const float c = u_in[idx];

    if ((i == 0u) | (j == 0u) | ((i + 1u) >= nx) | ((j + 1u) >= ny)) {
        u_out[idx] = c;
        return;
    }

    const float lr  = u_in[idx - 1u]  + u_in[idx + 1u];
    const float ud  = u_in[idx - nx]  + u_in[idx + nx];
    const float lap = (lr + ud) - (4.0f * c);

    u_out[idx] = fma(alpha, lap, c);
}
```