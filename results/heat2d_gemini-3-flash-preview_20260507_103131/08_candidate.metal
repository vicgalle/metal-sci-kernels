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

    // Bounds check to handle dispatches larger than the grid
    if (i >= NX || j >= NY) return;

    const uint idx = j * NX + i;

    // Dirichlet Boundary Conditions: Edge cells stay at their initial value.
    // Grouping i and j checks to allow the compiler to optimize branch logic.
    if (i == 0 || i == NX - 1 || j == 0 || j == NY - 1) {
        u_out[idx] = u_in[idx];
        return;
    }

    // Stencil loads: center and its 4 neighbors.
    // Using relative offsets to avoid re-calculating (j-1)*NX and (j+1)*NX.
    const float c = u_in[idx];
    const float l = u_in[idx - 1];
    const float r = u_in[idx + 1];
    const float d = u_in[idx - NX]; // Row above
    const float u = u_in[idx + NX]; // Row below

    // Discrete Laplacian calculation: (L + R + D + U - 4*C)
    // Structure additions to encourage Instruction-Level Parallelism (ILP).
    const float neighbors_sum = (l + r) + (d + u);
    const float laplacian = neighbors_sum - 4.0f * c;

    // Heat Equation Step: u_new = u + alpha * Laplacian
    // Use fma for precision and to potentially combine operations into one cycle.
    u_out[idx] = fma(alpha, laplacian, c);
}