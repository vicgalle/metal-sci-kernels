Applies a hybrid stencil optimization: small grids use a low-overhead cached path, large suitable 3D threadgroups stage `u_curr` in threadgroup memory, and other large layouts use SIMDgroup shuffles for intra-SIMD x/y/z neighbor reuse. This fixes the prior compile failure by removing the misplaced function attribute, while reducing redundant neighbor loads and keeping boundary handling deterministic.

```metal
#include <metal_stdlib>
using namespace metal;

kernel void wave_step(device const float *__restrict__ u_prev [[buffer(0)]],
                      device const float *__restrict__ u_curr [[buffer(1)]],
                      device       float *__restrict__ u_next [[buffer(2)]],
                      constant uint      &NX     [[buffer(3)]],
                      constant uint      &NY     [[buffer(4)]],
                      constant uint      &NZ     [[buffer(5)]],
                      constant float     &alpha  [[buffer(6)]],
                      uint3 gid  [[thread_position_in_grid]],
                      uint3 ltid [[thread_position_in_threadgroup]],
                      uint3 tgdim [[threads_per_threadgroup]],
                      uint  tidx [[thread_index_in_threadgroup]],
                      uint  lane [[thread_index_in_simdgroup]])
{
    threadgroup float tile[1024];

    const uint nx = NX;
    const uint ny = NY;
    const uint nz = NZ;

    const uint i = gid.x;
    const uint j = gid.y;
    const uint k = gid.z;

    const uint stride_y = nx;
    const uint stride_z = nx * ny;
    const uint idx = (k * ny + j) * nx + i;

    const uint tg_count = tgdim.x * tgdim.y * tgdim.z;

    // For the 64^3 case the hardware caches are usually enough; avoid
    // barriers/shared-memory overhead.  For large groups with real Z depth,
    // use threadgroup tiling.  Otherwise use a barrier-free SIMD shuffle path.
    const bool small_grid = (nx <= 96u) | (ny <= 96u) | (nz <= 96u);
    const bool use_tgm =
        (!small_grid) &
        (tg_count >= 128u) & (tg_count <= 1024u) &
        (tgdim.x >= 4u) & (tgdim.y >= 2u) & (tgdim.z >= 2u);

    if (small_grid) {
        if ((i >= nx) | (j >= ny) | (k >= nz)) return;

        const float c = u_curr[idx];

        const bool interior =
            (i != 0u) & (j != 0u) & (k != 0u) &
            (i < (nx - 1u)) & (j < (ny - 1u)) & (k < (nz - 1u));

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
        return;
    }

    if (!use_tgm) {
        const bool in_bounds = (i < nx) & (j < ny) & (k < nz);

        float c = 0.0f;
        if (in_bounds) {
            c = u_curr[idx];
        }

        const uint lane_u = lane;

        bool can_xm = false;
        bool can_xp = false;
        bool can_ym = false;
        bool can_yp = false;
        bool can_zm = false;
        bool can_zp = false;

        float sh_xm = 0.0f;
        float sh_xp = 0.0f;
        float sh_ym = 0.0f;
        float sh_yp = 0.0f;
        float sh_zm = 0.0f;
        float sh_zp = 0.0f;

        if (tgdim.x > 1u) {
            can_xm = (ltid.x != 0u) & (lane_u != 0u);
            can_xp = ((ltid.x + 1u) < tgdim.x) & ((lane_u + 1u) < 32u);

            const uint xm_lane = can_xm ? (lane_u - 1u) : lane_u;
            const uint xp_lane = can_xp ? (lane_u + 1u) : lane_u;

            sh_xm = simd_shuffle(c, ushort(xm_lane));
            sh_xp = simd_shuffle(c, ushort(xp_lane));
        }

        if ((tgdim.x < 32u) & (tgdim.y > 1u)) {
            const uint sx = tgdim.x;

            can_ym = (ltid.y != 0u) & (lane_u >= sx);
            can_yp = ((ltid.y + 1u) < tgdim.y) & ((lane_u + sx) < 32u);

            const uint ym_lane = can_ym ? (lane_u - sx) : lane_u;
            const uint yp_lane = can_yp ? (lane_u + sx) : lane_u;

            sh_ym = simd_shuffle(c, ushort(ym_lane));
            sh_yp = simd_shuffle(c, ushort(yp_lane));
        }

        const uint sxy = tgdim.x * tgdim.y;
        if ((sxy < 32u) & (tgdim.z > 1u)) {
            can_zm = (ltid.z != 0u) & (lane_u >= sxy);
            can_zp = ((ltid.z + 1u) < tgdim.z) & ((lane_u + sxy) < 32u);

            const uint zm_lane = can_zm ? (lane_u - sxy) : lane_u;
            const uint zp_lane = can_zp ? (lane_u + sxy) : lane_u;

            sh_zm = simd_shuffle(c, ushort(zm_lane));
            sh_zp = simd_shuffle(c, ushort(zp_lane));
        }

        if (!in_bounds) return;

        const bool interior =
            (i != 0u) & (j != 0u) & (k != 0u) &
            (i < (nx - 1u)) & (j < (ny - 1u)) & (k < (nz - 1u));

        if (!interior) {
            u_next[idx] = c;
            return;
        }

        float xm = sh_xm;
        float xp = sh_xp;
        float ym = sh_ym;
        float yp = sh_yp;
        float zm = sh_zm;
        float zp = sh_zp;

        if (!can_xm) xm = u_curr[idx - 1u];
        if (!can_xp) xp = u_curr[idx + 1u];
        if (!can_ym) ym = u_curr[idx - stride_y];
        if (!can_yp) yp = u_curr[idx + stride_y];
        if (!can_zm) zm = u_curr[idx - stride_z];
        if (!can_zp) zp = u_curr[idx + stride_z];

        const float lap = xm + xp + ym + yp + zm + zp - 6.0f * c;
        u_next[idx] = 2.0f * c - u_prev[idx] + alpha * lap;
        return;
    }

    // Large-grid path: stage the current threadgroup's 3D block of u_curr.
    const bool in_bounds = (i < nx) & (j < ny) & (k < nz);

    float c = 0.0f;
    if (in_bounds) {
        c = u_curr[idx];
    }

    tile[tidx] = c;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!in_bounds) return;

    const bool interior =
        (i != 0u) & (j != 0u) & (k != 0u) &
        (i < (nx - 1u)) & (j < (ny - 1u)) & (k < (nz - 1u));

    if (!interior) {
        u_next[idx] = c;
        return;
    }

    const uint sx  = tgdim.x;
    const uint sxy2 = tgdim.x * tgdim.y;

    const float xm = (ltid.x != 0u)            ? tile[tidx - 1u]    : u_curr[idx - 1u];
    const float xp = ((ltid.x + 1u) < tgdim.x) ? tile[tidx + 1u]    : u_curr[idx + 1u];

    const float ym = (ltid.y != 0u)            ? tile[tidx - sx]    : u_curr[idx - stride_y];
    const float yp = ((ltid.y + 1u) < tgdim.y) ? tile[tidx + sx]    : u_curr[idx + stride_y];

    const float zm = (ltid.z != 0u)            ? tile[tidx - sxy2]  : u_curr[idx - stride_z];
    const float zp = ((ltid.z + 1u) < tgdim.z) ? tile[tidx + sxy2]  : u_curr[idx + stride_z];

    const float lap = xm + xp + ym + yp + zm + zp - 6.0f * c;
    u_next[idx] = 2.0f * c - u_prev[idx] + alpha * lap;
}
```