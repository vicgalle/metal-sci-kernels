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
                      uint  tidx [[thread_index_in_threadgroup]])
                      [[max_total_threads_per_threadgroup(1024)]]
{
    threadgroup float tile[1024];

    const uint nx = NX;
    const uint ny = NY;
    const uint nz = NZ;

    const uint i = gid.x;
    const uint j = gid.y;
    const uint k = gid.z;

    const uint tg_count = tgdim.x * tgdim.y * tgdim.z;

    // For the small benchmark, or degenerate/very small threadgroups, avoid
    // the barrier and shared-memory traffic: the simple cached stencil wins.
    if ((nx <= 96u) || (ny <= 96u) || (nz <= 96u) ||
        (tg_count < 64u) || (tgdim.x < 4u) || (tgdim.y < 2u)) {

        if ((i >= nx) | (j >= ny) | (k >= nz)) return;

        const uint stride_y = nx;
        const uint stride_z = nx * ny;
        const uint idx = (k * ny + j) * nx + i;

        const bool interior =
            ((i - 1u) < (nx - 2u)) &
            ((j - 1u) < (ny - 2u)) &
            ((k - 1u) < (nz - 2u));

        const float c = u_curr[idx];

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

    // Large-grid path: stage this threadgroup's 3D block of u_curr.
    const bool in_bounds = (i < nx) & (j < ny) & (k < nz);

    const uint stride_y = nx;
    const uint stride_z = nx * ny;
    const uint idx = (k * ny + j) * nx + i;

    const bool interior =
        in_bounds &
        ((i - 1u) < (nx - 2u)) &
        ((j - 1u) < (ny - 2u)) &
        ((k - 1u) < (nz - 2u));

    float c = 0.0f;
    float p = 0.0f;

    if (in_bounds) {
        c = u_curr[idx];
        if (interior) {
            p = u_prev[idx];
        }
    }

    tile[tidx] = c;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!in_bounds) return;

    if (!interior) {
        u_next[idx] = c;
        return;
    }

    const uint sx  = tgdim.x;
    const uint sxy = tgdim.x * tgdim.y;

    const float xm = (ltid.x != 0u)                 ? tile[tidx - 1u]  : u_curr[idx - 1u];
    const float xp = ((ltid.x + 1u) < tgdim.x)      ? tile[tidx + 1u]  : u_curr[idx + 1u];

    const float ym = (ltid.y != 0u)                 ? tile[tidx - sx]  : u_curr[idx - stride_y];
    const float yp = ((ltid.y + 1u) < tgdim.y)      ? tile[tidx + sx]  : u_curr[idx + stride_y];

    const float zm = (ltid.z != 0u)                 ? tile[tidx - sxy] : u_curr[idx - stride_z];
    const float zp = ((ltid.z + 1u) < tgdim.z)      ? tile[tidx + sxy] : u_curr[idx + stride_z];

    const float lap = xm + xp + ym + yp + zm + zp - 6.0f * c;
    u_next[idx] = 2.0f * c - p + alpha * lap;
}