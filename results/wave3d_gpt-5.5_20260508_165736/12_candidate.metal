#include <metal_stdlib>
using namespace metal;

kernel void wave_step(device const float *__restrict__ u_prev [[buffer(0)]],
                      device const float *__restrict__ u_curr [[buffer(1)]],
                      device       float *__restrict__ u_next [[buffer(2)]],
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

    const uint idx = (k * ny + j) * nx + i;
    const float c = u_curr[idx];

    // Do the SIMD exchange before the boundary return so boundary lanes are
    // still active and can provide valid neighbor values to i==1 / i==NX-2.
    const ushort one = ushort(1);
    const uint  idx_l = simd_shuffle_up(idx, one);
    const uint  idx_r = simd_shuffle_down(idx, one);
    const float c_l   = simd_shuffle_up(c, one);
    const float c_r   = simd_shuffle_down(c, one);

    if (i == 0u || j == 0u || k == 0u ||
        i == nx - 1u || j == ny - 1u || k == nz - 1u) {
        u_next[idx] = c;
        return;
    }

    const uint stride_y = nx;
    const uint stride_z = nx * ny;

    float xm;
    if (idx_l == idx - 1u) {
        xm = c_l;
    } else {
        xm = u_curr[idx - 1u];
    }

    float xp;
    if (idx_r == idx + 1u) {
        xp = c_r;
    } else {
        xp = u_curr[idx + 1u];
    }

    const float ym = u_curr[idx - stride_y];
    const float yp = u_curr[idx + stride_y];
    const float zm = u_curr[idx - stride_z];
    const float zp = u_curr[idx + stride_z];
    const float prev = u_prev[idx];

    const float lap = (xm + xp) + (ym + yp) + (zm + zp) - 6.0f * c;
    u_next[idx] = fma(alpha, lap, (c + c) - prev);
}