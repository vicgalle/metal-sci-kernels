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

    // Guard for threads outside the defined grid
    if (i >= NX || j >= NY) return;

    const uint idx = j * NX + i;

    // Dirichlet Boundary Conditions: Edge cells stay at their initial value.
    if (i == 0 || j == 0 || i == NX - 1 || j == NY - 1) {
        u_out[idx] = u_in[idx];
        return;
    }

    // Stencil loads: center and its 4 neighbors.
    // Explicit row indexing helps the compiler maintain cache efficiency.
    const float c = u_in[idx];
    const float l = u_in[idx - 1];
    const float r = u_in[idx + 1];
    const float u = u_in[(j - 1) * NX + i];
    const float d = u_in[(j + 1) * NX + i];

    // Discrete Laplacian: (L + R + U + D - 4*C)
    // Formula: u_new = u + alpha * Laplacian
    // Using Fused Multiply-Add (FMA) for numerical precision and performance.
    const float laplacian = (l + r + u + d) - 4.0f * c;
    u_out[idx] = fma(alpha, laplacian, c);
}