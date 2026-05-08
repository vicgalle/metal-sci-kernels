#include <metal_stdlib>
using namespace metal;

kernel void heat_step(device const float *u_in  [[buffer(0)]],
                      device       float *u_out [[buffer(1)]],
                      constant uint      &NX    [[buffer(2)]],
                      constant uint      &NY    [[buffer(3)]],
                      constant float     &alpha [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]],
                      uint2 tid [[thread_position_in_threadgroup]],
                      uint2 tpg [[threads_per_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]]) {
    const uint i  = gid.x;
    const uint j  = gid.y;
    const uint nx = NX;
    const uint ny = NY;

    if (i >= nx || j >= ny) return;

    const uint idx = j * nx + i;

    // For small grids, the incumbent-style scalar path avoids shuffle overhead.
    if (nx <= 256u || ny <= 256u || tpg.x < 2u) {
        if (i == 0u || j == 0u || i == nx - 1u || j == ny - 1u) {
            u_out[idx] = u_in[idx];
            return;
        }

        const float c = u_in[idx];
        const float l = u_in[idx - 1u];
        const float r = u_in[idx + 1u];
        const float d = u_in[idx - nx];
        const float u = u_in[idx + nx];

        u_out[idx] = c + alpha * (l + r + d + u - 4.0f * c);
        return;
    }

    // All in-grid boundary lanes participate so neighboring interior lanes can
    // read their boundary value via SIMD shuffle.
    const float c = u_in[idx];

    const float l_shuf = simd_shuffle_up(c, ushort(1));
    const float r_shuf = simd_shuffle_down(c, ushort(1));

    const bool have_l = (tid.x != 0u) && (lane != 0);
    const bool have_r = ((tid.x + 1u) < tpg.x) && (lane < 31);

    if (i == 0u || j == 0u || i == nx - 1u || j == ny - 1u) {
        u_out[idx] = c;
        return;
    }

    float l = l_shuf;
    if (!have_l) {
        l = u_in[idx - 1u];
    }

    float r = r_shuf;
    if (!have_r) {
        r = u_in[idx + 1u];
    }

    const float d = u_in[idx - nx];
    const float u = u_in[idx + nx];

    u_out[idx] = c + alpha * (l + r + d + u - 4.0f * c);
}