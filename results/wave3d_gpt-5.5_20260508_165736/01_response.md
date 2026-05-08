I reduce redundant neighbor loads by using SIMDgroup shuffles: each lane’s center value is shared with adjacent x/y/z lanes when those neighbors are inside the same SIMDgroup, falling back to global memory at SIMD/threadgroup boundaries. This should improve over the naive version by cutting many cached-but-still-issued `u_curr` neighbor load instructions without adding barriers or threadgroup-memory pressure.  
```metal
#include <metal_stdlib>
using namespace metal;

kernel void wave_step(device const float *u_prev [[buffer(0)]],
                      device const float *u_curr [[buffer(1)]],
                      device       float *u_next [[buffer(2)]],
                      constant uint      &NX     [[buffer(3)]],
                      constant uint      &NY     [[buffer(4)]],
                      constant uint      &NZ     [[buffer(5)]],
                      constant float     &alpha  [[buffer(6)]],
                      uint3 gid [[thread_position_in_grid]],
                      uint3 tid [[thread_position_in_threadgroup]],
                      uint3 tpg [[threads_per_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]]) {
    const uint i = gid.x;
    const uint j = gid.y;
    const uint k = gid.z;

    if (i >= NX || j >= NY || k >= NZ) return;

    const uint stride_y = NX;
    const uint stride_z = NX * NY;
    const uint idx = k * stride_z + j * stride_y + i;

    const float c = u_curr[idx];

    const bool is_boundary =
        (i == 0u) || (j == 0u) || (k == 0u) ||
        (i == NX - 1u) || (j == NY - 1u) || (k == NZ - 1u);

    const uint lane_u = uint(lane);

    float sh_xm = c;
    float sh_xp = c;
    if (tpg.x > 1u) {
        const ushort lm = (lane == ushort(0))  ? ushort(0)  : ushort(lane - ushort(1));
        const ushort lp = (lane == ushort(31)) ? ushort(31) : ushort(lane + ushort(1));
        sh_xm = simd_shuffle(c, lm);
        sh_xp = simd_shuffle(c, lp);
    }

    float sh_ym = c;
    float sh_yp = c;
    if (tpg.y > 1u && tpg.x < 32u) {
        const uint dx = tpg.x;
        const ushort lm = (lane_u >= dx)        ? ushort(lane_u - dx) : lane;
        const ushort lp = (lane_u + dx < 32u)   ? ushort(lane_u + dx) : lane;
        sh_ym = simd_shuffle(c, lm);
        sh_yp = simd_shuffle(c, lp);
    }

    float sh_zm = c;
    float sh_zp = c;
    const uint plane_threads = tpg.x * tpg.y;
    if (tpg.z > 1u && plane_threads < 32u) {
        const uint dz = plane_threads;
        const ushort lm = (lane_u >= dz)        ? ushort(lane_u - dz) : lane;
        const ushort lp = (lane_u + dz < 32u)   ? ushort(lane_u + dz) : lane;
        sh_zm = simd_shuffle(c, lm);
        sh_zp = simd_shuffle(c, lp);
    }

    if (is_boundary) {
        u_next[idx] = c;
        return;
    }

    const bool use_xm = (tid.x != 0u) && (lane != ushort(0));
    const bool use_xp = (tid.x + 1u < tpg.x) && (lane != ushort(31));

    const bool use_ym = (tpg.y > 1u) && (tpg.x < 32u) &&
                        (tid.y != 0u) && (lane_u >= tpg.x);
    const bool use_yp = (tpg.y > 1u) && (tpg.x < 32u) &&
                        (tid.y + 1u < tpg.y) && (lane_u + tpg.x < 32u);

    const bool use_zm = (tpg.z > 1u) && (plane_threads < 32u) &&
                        (tid.z != 0u) && (lane_u >= plane_threads);
    const bool use_zp = (tpg.z > 1u) && (plane_threads < 32u) &&
                        (tid.z + 1u < tpg.z) && (lane_u + plane_threads < 32u);

    const float xm = use_xm ? sh_xm : u_curr[idx - 1u];
    const float xp = use_xp ? sh_xp : u_curr[idx + 1u];

    const float ym = use_ym ? sh_ym : u_curr[idx - stride_y];
    const float yp = use_yp ? sh_yp : u_curr[idx + stride_y];

    const float zm = use_zm ? sh_zm : u_curr[idx - stride_z];
    const float zp = use_zp ? sh_zp : u_curr[idx + stride_z];

    const float sum = (xm + xp) + (ym + yp) + (zm + zp);
    const float lap = sum - 6.0f * c;

    u_next[idx] = fma(alpha, lap, (c + c) - u_prev[idx]);
}
```