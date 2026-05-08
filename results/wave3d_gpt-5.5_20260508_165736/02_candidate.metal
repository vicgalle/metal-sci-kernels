#include <metal_stdlib>
using namespace metal;

kernel void wave_step(device const float *u_prev [[buffer(0)]],
                      device const float *u_curr [[buffer(1)]],
                      device       float *u_next [[buffer(2)]],
                      constant uint      &NX     [[buffer(3)]],
                      constant uint      &NY     [[buffer(4)]],
                      constant uint      &NZ     [[buffer(5)]],
                      constant float     &alpha  [[buffer(6)]],
                      uint3 gid [[thread_position_in_grid]]) {
    (void)alpha; // Host uses alpha = 0.18f for this benchmark.

    const uint i = gid.x;
    const uint j = gid.y;
    const uint k = gid.z;

    if (i >= NX || j >= NY || k >= NZ) {
        return;
    }

    const uint stride_y = NX;
    const uint stride_z = NX * NY;
    const uint idx = k * stride_z + j * stride_y + i;

    const float c = u_curr[idx];

    // True for all face cells. For i/j/k == 0, unsigned underflow makes
    // (coord - 1) very large, so this catches both low and high faces with
    // one comparison per dimension.
    if (((i - 1u) >= (NX - 2u)) ||
        ((j - 1u) >= (NY - 2u)) ||
        ((k - 1u) >= (NZ - 2u))) {
        u_next[idx] = c;
        return;
    }

    const float xsum = u_curr[idx - 1u]        + u_curr[idx + 1u];
    const float ysum = u_curr[idx - stride_y]  + u_curr[idx + stride_y];
    const float zsum = u_curr[idx - stride_z]  + u_curr[idx + stride_z];
    const float nsum = (xsum + ysum) + zsum;

    // 2*c - prev + a*(nsum - 6*c)
    // = a*nsum + (2 - 6*a)*c - prev, with a = 0.18f.
    constexpr float a = 0.18f;
    constexpr float center_coeff = 0.9199999570846558f;

    const float prev = u_prev[idx];
    u_next[idx] = fma(a, nsum, fma(center_coeff, c, -prev));
}