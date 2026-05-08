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

    // Dirichlet boundary conditions: edge cells stay at their initial value.
    if (i == 0 || j == 0 || i == nx - 1 || j == ny - 1) {
        u_out[idx] = u_in[idx];
        return;
    }

    // Monotonically ordered loads to favor hardware cache prefetching
    float d = u_in[idx - nx];
    float l = u_in[idx - 1];
    float c = u_in[idx];
    float r = u_in[idx + 1];
    float u = u_in[idx + nx];

    float a = alpha;
    
    // Precompute center weight to save a multiply-subtract per cell
    float c0 = 1.0f - 4.0f * a;

    // FMA optimization: c * c0 + a * sum_of_neighbors
    u_out[idx] = c * c0 + a * (l + r + d + u);
}