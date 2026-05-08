#include <metal_stdlib>
using namespace metal;

kernel void heat_step(device const float *u_in  [[buffer(0)]],
                      device       float *u_out [[buffer(1)]],
                      constant uint      &NX    [[buffer(2)]],
                      constant uint      &NY    [[buffer(3)]],
                      constant float     &alpha [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x;
    const uint j = gid.y;

    // Grid boundary guard
    if (i >= NX || j >= NY) return;

    // Compute linear index once for row-major storage
    const uint idx = j * NX + i;
    const float center_val = u_in[idx];

    // Dirichlet Boundary Conditions: Edge cells stay constant
    // j=0 (top), j=NY-1 (bottom), i=0 (left), i=NX-1 (right)
    if (i == 0 || i == NX - 1 || j == 0 || j == NY - 1) {
        u_out[idx] = center_val;
        return;
    }

    // 5-point stencil: Left, Right, Up, Down neighbors
    // Vertical neighbors (idx - NX, idx + NX) are typically fetched from L2 cache
    const float l = u_in[idx - 1];
    const float r = u_in[idx + 1];
    const float d = u_in[idx - NX];
    const float u = u_in[idx + NX];

    // Optimized math: u_new = (1.0 - 4.0 * alpha) * c + alpha * (l + r + u + d)
    // This formulation reduces the instruction count compared to the standard Laplacian.
    const float beta = 1.0f - 4.0f * alpha;
    u_out[idx] = beta * center_val + alpha * (l + r + d + u);
}