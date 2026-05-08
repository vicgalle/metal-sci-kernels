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
    uint nx = NX;
    uint ny = NY;

    if (i >= nx || j >= ny) return;

    uint idx = j * nx + i;
    
    // Load center value immediately; this is used by both boundary and interior cells.
    // Hoisting this load guarantees uniform memory access across the SIMD group.
    float c = u_in[idx];

    // Dirichlet boundary conditions: edge cells stay at their initial value.
    if (i == 0 || j == 0 || i == nx - 1 || j == ny - 1) {
        u_out[idx] = c;
        return;
    }

    // Fetch neighbors using simple offset arithmetic rather than full 2D index recalculation.
    // This strictly reduces integer ALU operations.
    float l = u_in[idx - 1];
    float r = u_in[idx + 1];
    float d = u_in[idx - nx];
    float u = u_in[idx + nx];

    // Grouped additions `(l + r) + (d + u)` reduce latency via instruction-level parallelism.
    // The FMA intrinsic efficiently fuses the final scaling and addition.
    u_out[idx] = fma(alpha, (l + r) + (d + u) - 4.0f * c, c);
}