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

    // Bounds check for the dispatch grid
    if (i >= NX || j >= NY) return;

    const uint idx = j * NX + i;

    // Dirichlet Boundary Conditions: Edge cells stay at their initial value.
    if (i == 0 || j == 0 || i == NX - 1 || j == NY - 1) {
        u_out[idx] = u_in[idx];
        return;
    }

    // Load center and its 4 neighbors
    // Note: L1/L2 cache effectively handles the stencil overlaps between threads.
    const float c = u_in[idx];
    const float l = u_in[idx - 1];
    const float r = u_in[idx + 1];
    const float n = u_in[idx - NX];
    const float s = u_in[idx + NX];

    // Discrete Laplacian: L + R + N + S - 4*C
    // Using Fused Multiply-Add (FMA) for the update step: u_new = alpha * Laplacian + center
    const float laplacian = (l + r + n + s) - 4.0f * c;
    u_out[idx] = fma(alpha, laplacian, c);
}