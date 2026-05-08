#include <metal_stdlib>
using namespace metal;

kernel void wave_step(device const float *u_prev [[buffer(0)]],
                      device const float *u_curr [[buffer(1)]],
                      device       float *u_next [[buffer(2)]],
                      constant uint      &NX     [[buffer(3)]],
                      constant uint      &NY     [[buffer(4)]],
                      constant uint      &NZ     [[buffer(5)]],
                      constant float     &alpha  [[buffer(6)]],
                      uint3 gid [[thread_position_in_grid]])
{
    const uint i = gid.x;
    const uint j = gid.y;
    const uint k = gid.z;

    const uint nx = NX;
    const uint ny = NY;
    const uint nz = NZ;

    if (i >= nx || j >= ny || k >= nz) {
        return;
    }

    device const float *__restrict__ prev = u_prev;
    device const float *__restrict__ curr = u_curr;
    device       float *__restrict__ next = u_next;

    const uint idx = (k * ny + j) * nx + i;
    const float c = curr[idx];

    if ((i - 1u) >= (nx - 2u) ||
        (j - 1u) >= (ny - 2u) ||
        (k - 1u) >= (nz - 2u)) {
        next[idx] = c;
        return;
    }

    const uint stride_y = nx;
    const uint stride_z = nx * ny;

    const float neighbor_sum =
        (curr[idx - 1u]        + curr[idx + 1u]) +
        (curr[idx - stride_y]  + curr[idx + stride_y]) +
        (curr[idx - stride_z]  + curr[idx + stride_z]);

    constexpr float a = 0.18f;
    constexpr float center_coeff = 2.0f - 6.0f * a;

    const float p = prev[idx];
    next[idx] = fma(a, neighbor_sum, fma(center_coeff, c, -p));
}