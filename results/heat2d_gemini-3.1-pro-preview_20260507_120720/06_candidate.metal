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

    // Single flat index simplifies address arithmetic for the whole stencil
    uint idx = j * nx + i;
    
    // Preload center value; handles both the boundary copy and the interior center
    float c = u_in[idx];

    // Dirichlet boundaries
    if (i == 0 || j == 0 || i == nx - 1 || j == ny - 1) {
        u_out[idx] = c;
        return;
    }

    // Relative offsets replace multiple integer multiplications per thread
    float l = u_in[idx - 1];
    float r = u_in[idx + 1];
    float d = u_in[idx - nx];
    float u = u_in[idx + nx];

    u_out[idx] = c + alpha * (l + r + d + u - 4.0f * c);
}