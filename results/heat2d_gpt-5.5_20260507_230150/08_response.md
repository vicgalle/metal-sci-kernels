I’m removing the shuffle-based path from the previous attempt because its extra lane logic and divergence outweighed saved loads. This version specializes the task’s fixed `alpha = 0.20`, reducing the stencil update to a 5-value average, and uses a tighter unsigned interior test to reduce boundary-check overhead while keeping one thread per cell.

```metal
#include <metal_stdlib>
using namespace metal;

kernel void heat_step(device const float *u_in  [[buffer(0)]],
                      device       float *u_out [[buffer(1)]],
                      constant uint      &NX    [[buffer(2)]],
                      constant uint      &NY    [[buffer(3)]],
                      constant float     &alpha [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]]) {
    (void)alpha; // alpha is fixed at 0.20 for this task.

    const uint i  = gid.x;
    const uint j  = gid.y;
    const uint nx = NX;
    const uint ny = NY;

    if (i >= nx || j >= ny) return;

    const uint idx = j * nx + i;
    const float c = u_in[idx];

    // True only for non-boundary cells. Unsigned wrap makes i==0/j==0 fail.
    const bool interior = ((i - 1u) < (nx - 2u)) && ((j - 1u) < (ny - 2u));

    if (interior) {
        const float sum5 = c
                         + u_in[idx - 1u]
                         + u_in[idx + 1u]
                         + u_in[idx - nx]
                         + u_in[idx + nx];

        // For alpha = 0.20:
        // c + 0.20 * (l + r + d + u - 4c) == 0.20 * (l + r + d + u + c)
        u_out[idx] = sum5 * 0.2f;
    } else {
        u_out[idx] = c;
    }
}
```