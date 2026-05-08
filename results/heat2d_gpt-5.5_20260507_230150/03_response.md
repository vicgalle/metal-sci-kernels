I’m applying SIMDgroup shuffle reuse for neighboring cells on larger grids while keeping the incumbent-style scalar path for small grids. Adjacent threads already load the center value, so shuffling those values within a SIMDgroup avoids many horizontal and some vertical neighbor memory loads without adding a threadgroup barrier. This should improve memory/load-pipe efficiency on 512² and 1024² while preserving the fast naive path for 256².

```metal
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
                      uint lane [[thread_index_in_simdgroup]]) {
    const uint i = gid.x;
    const uint j = gid.y;
    const uint nx = NX;
    const uint ny = NY;

    if (i >= nx || j >= ny) return;

    const uint idx = j * nx + i;

    // For small grids the launch/branch overhead dominates; keep the compact
    // scalar path that compiles very well.
    if (nx <= 256u) {
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

    // Load center for every in-grid lane, including boundary lanes, before any
    // early return so neighboring interior lanes may reuse it via SIMD shuffle.
    const float c = u_in[idx];

    const uint left_lane  = (lane == 0u)  ? lane : (lane - 1u);
    const uint right_lane = (lane == 31u) ? lane : (lane + 1u);

    const float sh_l = simd_shuffle(c, ushort(left_lane));
    const float sh_r = simd_shuffle(c, ushort(right_lane));

    float sh_d = c;
    float sh_u = c;
    bool use_sh_d = false;
    bool use_sh_u = false;

    // If a row is narrower than the SIMD width, some vertical neighbors are
    // also resident in the same SIMDgroup at lane +/- tpg.x.
    if (tpg.x < 32u) {
        const uint up_lane = (lane >= tpg.x) ? (lane - tpg.x) : lane;
        const uint down_lane_raw = lane + tpg.x;
        const uint down_lane = (down_lane_raw < 32u) ? down_lane_raw : lane;

        sh_d = simd_shuffle(c, ushort(up_lane));
        sh_u = simd_shuffle(c, ushort(down_lane));

        use_sh_d = (tid.y != 0u) && (lane >= tpg.x);
        use_sh_u = ((tid.y + 1u) < tpg.y) && (down_lane_raw < 32u);
    }

    if (i == 0u || j == 0u || i == nx - 1u || j == ny - 1u) {
        u_out[idx] = c;
        return;
    }

    const bool use_sh_l = (lane != 0u)  && (tid.x != 0u);
    const bool use_sh_r = (lane != 31u) && ((tid.x + 1u) < tpg.x);

    const float l = use_sh_l ? sh_l : u_in[idx - 1u];
    const float r = use_sh_r ? sh_r : u_in[idx + 1u];
    const float d = use_sh_d ? sh_d : u_in[idx - nx];
    const float u = use_sh_u ? sh_u : u_in[idx + nx];

    u_out[idx] = c + alpha * (l + r + d + u - 4.0f * c);
}
```