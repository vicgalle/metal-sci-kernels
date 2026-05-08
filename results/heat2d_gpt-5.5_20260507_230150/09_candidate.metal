#include <metal_stdlib>
using namespace metal;

kernel void heat_step(device const float *u_in  [[buffer(0)]],
                      device       float *u_out [[buffer(1)]],
                      constant uint      &NX    [[buffer(2)]],
                      constant uint      &NY    [[buffer(3)]],
                      constant float     &alpha [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]],
                      uint2 lid [[thread_position_in_threadgroup]],
                      uint2 tpg [[threads_per_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]]) {
    const uint i  = gid.x;
    const uint j  = gid.y;
    const uint nx = NX;
    const uint ny = NY;

    if (i >= nx || j >= ny) return;

    const uint idx = j * nx + i;
    const float c = u_in[idx];

    // All in-grid lanes load c and execute these shuffles before boundary lanes
    // copy out, so interior cells can safely read boundary neighbors via lanes.
    const uint lane_u = uint(lane);

    const float sh_l = simd_shuffle(c, ushort((lane_u > 0u)  ? (lane_u - 1u) : 0u));
    const float sh_r = simd_shuffle(c, ushort((lane_u < 31u) ? (lane_u + 1u) : 31u));

    const uint row_w = tpg.x;
    float sh_up = c;
    float sh_dn = c;

    // If a SIMDgroup contains multiple threadgroup rows, vertical neighbors may
    // also be in registers.  Otherwise these remain unused and we fall back.
    if (row_w < 32u) {
        const uint up_lane = (lane_u >= row_w) ? (lane_u - row_w) : lane_u;
        const uint dn_lane = ((lane_u + row_w) < 32u) ? (lane_u + row_w) : lane_u;
        sh_up = simd_shuffle(c, ushort(up_lane));
        sh_dn = simd_shuffle(c, ushort(dn_lane));
    }

    if (i == 0u || j == 0u || i + 1u >= nx || j + 1u >= ny) {
        u_out[idx] = c;
        return;
    }

    float l;
    if (lid.x != 0u && lane_u != 0u) {
        l = sh_l;
    } else {
        l = u_in[idx - 1u];
    }

    float r;
    if ((lid.x + 1u) < row_w && lane_u != 31u) {
        r = sh_r;
    } else {
        r = u_in[idx + 1u];
    }

    float upv;
    if (lid.y != 0u && lane_u >= row_w) {
        upv = sh_up;
    } else {
        upv = u_in[idx - nx];
    }

    float dnv;
    if ((lid.y + 1u) < tpg.y && (lane_u + row_w) < 32u) {
        dnv = sh_dn;
    } else {
        dnv = u_in[idx + nx];
    }

    u_out[idx] = c + alpha * (l + r + upv + dnv - 4.0f * c);
}