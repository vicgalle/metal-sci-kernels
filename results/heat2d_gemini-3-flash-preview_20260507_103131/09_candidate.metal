#include <metal_stdlib>
using namespace metal;

[[kernel]]
[[max_total_threads_per_threadgroup(256)]]
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

    // Dirichlet Boundary Conditions: i=0, i=NX-1, j=0, or j=NY-1.
    // Using unsigned underflow trick: (0 - 1) becomes UINT_MAX, 
    // which is greater than (NX - 2).
    if ((i - 1) >= (NX - 2) || (j - 1) >= (NY - 2)) {
        u_out[idx] = u_in[idx];
    } else {
        // Interior: 5-point stencil
        // Neighbors: Left, Right, Down (above row), Up (below row)
        const float c = u_in[idx];
        const float l = u_in[idx - 1];
        const float r = u_in[idx + 1];
        const float d = u_in[idx - NX];
        const float u = u_in[idx + NX];

        // u_new = u + alpha * (L + R + D + U - 4*C)
        // This arrangement is optimized for memory-bound stencil performance.
        u_out[idx] = c + alpha * (l + r + d + u - 4.0f * c);
    }
}