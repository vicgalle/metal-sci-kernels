#include <metal_stdlib>
using namespace metal;

kernel void wave_step(device const float *u_prev [[buffer(0)]],
                      device const float *u_curr [[buffer(1)]],
                      device       float *u_next [[buffer(2)]],
                      constant uint      &NX     [[buffer(3)]],
                      constant uint      &NY     [[buffer(4)]],
                      constant uint      &NZ     [[buffer(5)]],
                      constant float     &alpha  [[buffer(6)]],
                      uint3 gid  [[thread_position_in_grid]],
                      uint3 tid  [[thread_position_in_threadgroup]],
                      uint3 tpg  [[threads_per_threadgroup]],
                      uint  lane [[thread_index_in_simdgroup]]) {
    const uint i = gid.x;
    const uint j = gid.y;
    const uint k = gid.z;

    const bool in_bounds = (i < NX) && (j < NY) && (k < NZ);

    const uint stride_y = NX;
    const uint stride_z = NX * NY;
    const uint idx = k * stride_z + j * stride_y + i;

    float c = 0.0f;
    if (in_bounds) {
        c = u_curr[idx];
    }

    // Compute same-SIMD candidate neighbor values before boundary returns so
    // boundary lanes can still provide their u_curr value to adjacent interior
    // lanes. Values are only used when the source lane is known to correspond
    // to the desired grid neighbor; otherwise we fall back to a global load.
    float sxm = c;
    float sxp = c;
    if (tpg.x > 1u) {
        sxm = simd_shuffle(c, ushort((lane + 31u) & 31u));
        sxp = simd_shuffle(c, ushort((lane + 1u) & 31u));
    }

    float sym = c;
    float syp = c;
    const uint row_delta = tpg.x;
    if (row_delta < 32u) {
        const uint lane_ym = (lane >= row_delta) ? (lane - row_delta) : lane;
        const uint lane_yp = ((lane + row_delta) < 32u) ? (lane + row_delta) : lane;
        sym = simd_shuffle(c, ushort(lane_ym));
        syp = simd_shuffle(c, ushort(lane_yp));
    }

    if (!in_bounds) {
        return;
    }

    if (i == 0u || j == 0u || k == 0u ||
        i == NX - 1u || j == NY - 1u || k == NZ - 1u) {
        u_next[idx] = c;
        return;
    }

    float xm;
    if (tid.x > 0u && lane > 0u) {
        xm = sxm;
    } else {
        xm = u_curr[idx - 1u];
    }

    float xp;
    if ((tid.x + 1u < tpg.x) && (lane + 1u < 32u)) {
        xp = sxp;
    } else {
        xp = u_curr[idx + 1u];
    }

    float ym;
    if ((row_delta < 32u) && (tid.y > 0u) && (lane >= row_delta)) {
        ym = sym;
    } else {
        ym = u_curr[idx - stride_y];
    }

    float yp;
    if ((row_delta < 32u) && (tid.y + 1u < tpg.y) && (lane + row_delta < 32u)) {
        yp = syp;
    } else {
        yp = u_curr[idx + stride_y];
    }

    const float zm = u_curr[idx - stride_z];
    const float zp = u_curr[idx + stride_z];

    const float lap = xm + xp + ym + yp + zm + zp - 6.0f * c;
    u_next[idx] = 2.0f * c - u_prev[idx] + alpha * lap;
}