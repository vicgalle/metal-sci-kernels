#include <metal_stdlib>
using namespace metal;

kernel void heat_step(device const float *u_in  [[buffer(0)]],
                      device       float *u_out [[buffer(1)]],
                      constant uint      &NX    [[buffer(2)]],
                      constant uint      &NY    [[buffer(3)]],
                      constant float     &alpha [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]]) {
    const uint i  = gid.x;
    const uint j  = gid.y;
    const uint nx = NX;
    const uint ny = NY;

    if (i >= nx || j >= ny) {
        return;
    }

    const uint idx = j * nx + i;
    const float c = u_in[idx];

    // Boundary or degenerate dimensions: Dirichlet copy-through.
    if ((i - 1u) >= (nx - 2u) || (j - 1u) >= (ny - 2u)) {
        u_out[idx] = c;
        return;
    }

    // Fast path for the configured benchmark alpha = 0.20:
    // c + 0.2 * (neighbors - 4c) == 0.2 * (c + neighbors).
    if (alpha == 0.2f) {
        const float s = (((c + u_in[idx - 1u]) + u_in[idx + 1u])
                         + u_in[idx - nx]) + u_in[idx + nx];
        u_out[idx] = 0.2f * s;
        return;
    }

    const float l = u_in[idx - 1u];
    const float r = u_in[idx + 1u];
    const float u = u_in[idx - nx];
    const float d = u_in[idx + nx];
    const float nbr = (l + r) + (u + d);

    u_out[idx] = fma(alpha, nbr - 4.0f * c, c);
}