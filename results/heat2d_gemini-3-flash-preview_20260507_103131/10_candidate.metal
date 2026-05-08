#include <metal_stdlib>
using namespace metal;

[[kernel]]
void heat_step(device const float *u_in  [[buffer(0)]],
                      device       float *u_out [[buffer(1)]],
                      constant uint      &NX    [[buffer(2)]],
                      constant uint      &NY    [[buffer(3)]],
                      constant float     &alpha [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x;
    const uint j = gid.y;

    // Grid bounds check
    if (i >= NX || j >= NY) return;

    const uint idx = j * NX + i;
    const float center = u_in[idx];

    // Dirichlet Boundary Conditions: i=0, i=NX-1, j=0, or j=NY-1.
    // Early return for boundary threads minimizes branch divergence and overhead.
    if (i == 0 || i == NX - 1 || j == 0 || j == NY - 1) {
        u_out[idx] = center;
        return;
    }

    // Interior: 5-point stencil
    // Neighbors: Left, Right, Down (previous row), Up (next row)
    const float l = u_in[idx - 1];
    const float r = u_in[idx + 1];
    const float d = u_in[idx - NX];
    const float u = u_in[idx + NX];

    // u_new = u + alpha * (L + R + D + U - 4*C)
    // Balanced tree for neighbor summation assists instruction scheduling.
    u_out[idx] = center + alpha * ((l + r) + (d + u) - 4.0f * center);
}