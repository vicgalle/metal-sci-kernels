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
                      uint lane [[thread_index_in_simdgroup]]) {
    const uint i  = gid.x;
    const uint j  = gid.y;
    const uint nx = NX;
    const uint ny = NY;

    if (i >= nx || j >= ny) return;

    const uint idx = j * nx + i;
    const float c = u_in[idx];

    // All valid lanes, including boundary lanes, participate so neighboring
    // interior lanes can read boundary values through the SIMD shuffle.
    const float l_sh = simd_shuffle_up(c, (ushort)1);
    const float r_sh = simd_shuffle_down(c, (ushort)1);

    const uint tx = tpg.x;
    const bool do_v_shuffle = (tx <= 16u) && (tpg.y > 1u);

    float u_sh = c;
    float d_sh = c;
    if (do_v_shuffle) {
        const uint up_lane_u = (lane >= tx) ? (lane - tx) : lane;
        const uint dn_lane_u = ((lane + tx) < 32u) ? (lane + tx) : lane;
        u_sh = simd_shuffle(c, (ushort)up_lane_u);
        d_sh = simd_shuffle(c, (ushort)dn_lane_u);
    }

    if (i == 0u || j == 0u || i + 1u >= nx || j + 1u >= ny) {
        u_out[idx] = c;
        return;
    }

    float l = l_sh;
    if (lane == 0u || lid.x == 0u) {
        l = u_in[idx - 1u];
    }

    float r = r_sh;
    if (lane == 31u || (lid.x + 1u) >= tx) {
        r = u_in[idx + 1u];
    }

    float u = u_sh;
    if (!do_v_shuffle || lane < tx || lid.y == 0u) {
        u = u_in[idx - nx];
    }

    float d = d_sh;
    if (!do_v_shuffle || (lane + tx) >= 32u || (lid.y + 1u) >= tpg.y) {
        d = u_in[idx + nx];
    }

    const float sum = (l + r) + (u + d);
    u_out[idx] = c + alpha * (sum - 4.0f * c);
}