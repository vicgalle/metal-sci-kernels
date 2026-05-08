#include <metal_stdlib>
using namespace metal;

kernel void wave_step(device const float * __restrict u_prev [[buffer(0)]],
                      device const float * __restrict u_curr [[buffer(1)]],
                      device       float * __restrict u_next [[buffer(2)]],
                      constant uint      &NX     [[buffer(3)]],
                      constant uint      &NY     [[buffer(4)]],
                      constant uint      &NZ     [[buffer(5)]],
                      constant float     &alpha  [[buffer(6)]],
                      uint3 gid [[thread_position_in_grid]]) {
    const uint i = gid.x;
    const uint j = gid.y;
    const uint k = gid.z;

    const uint nx = NX;
    const uint ny = NY;
    const uint nz = NZ;

    if (i >= nx || j >= ny || k >= nz) {
        return;
    }

    const uint stride_y = nx;
    const uint stride_z = nx * ny;
    const uint idx = k * stride_z + j * stride_y + i;

    const float c = u_curr[idx];

    // Interior iff coordinate is in [1, dim-2].  This single unsigned-range
    // test per axis covers both Dirichlet faces, including tiny dimensions.
    const bool interior =
        ((i - 1u) < (nx - 2u)) &
        ((j - 1u) < (ny - 2u)) &
        ((k - 1u) < (nz - 2u));

    if (!interior) {
        u_next[idx] = c;
        return;
    }

    const float xm = u_curr[idx - 1u];
    const float xp = u_curr[idx + 1u];
    const float ym = u_curr[idx - stride_y];
    const float yp = u_curr[idx + stride_y];
    const float zm = u_curr[idx - stride_z];
    const float zp = u_curr[idx + stride_z];

    const float lap = xm + xp + ym + yp + zm + zp - 6.0f * c;
    u_next[idx] = 2.0f * c - u_prev[idx] + alpha * lap;
}