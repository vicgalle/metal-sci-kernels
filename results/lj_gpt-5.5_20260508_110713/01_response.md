Applies two main optimizations: cheaper minimum-image wrapping via conditional single-period correction, and conservative per-neighbour-cell bounding-box culling before scanning occupants. This avoids expensive per-pair `round` calls and skips many edge/corner neighbour cells that cannot contain particles within the cutoff, reducing both math and memory traffic while preserving one-thread-per-particle semantics.  

```metal
#include <metal_stdlib>
using namespace metal;

static inline float3 lj_wrap_pos(float3 r, float L, float invL) {
    return r - L * floor(r * invL);
}

static inline uint lj_cell_coord(float x, float inv_cell, uint M) {
    return min(uint(x * inv_cell), M - 1u);
}

static inline float3 lj_min_image_fast(float3 d, float L, float halfL) {
    float3 z = float3(0.0f);
    d -= select(z, float3(L),  d >  float3(halfL));
    d += select(z, float3(L),  d < -float3(halfL));
    return d;
}

kernel void lj_clear_cells(
    device atomic_uint *cell_count [[buffer(0)]],
    constant uint      &M3         [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= M3) return;
    atomic_store_explicit(&cell_count[gid], 0u, memory_order_relaxed);
}

kernel void lj_build_cells(
    device const float4 *pos          [[buffer(0)]],
    device atomic_uint  *cell_count   [[buffer(1)]],
    device       uint   *cell_list    [[buffer(2)]],
    constant uint        &N           [[buffer(3)]],
    constant uint        &M           [[buffer(4)]],
    constant float       &L           [[buffer(5)]],
    constant uint        &MAX_PER_CELL[[buffer(6)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= N) return;

    float invL     = 1.0f / L;
    float inv_cell = float(M) * invL;

    float3 r = lj_wrap_pos(pos[i].xyz, L, invL);

    uint cx = lj_cell_coord(r.x, inv_cell, M);
    uint cy = lj_cell_coord(r.y, inv_cell, M);
    uint cz = lj_cell_coord(r.z, inv_cell, M);

    uint cell = (cz * M + cy) * M + cx;
    uint slot = atomic_fetch_add_explicit(&cell_count[cell], 1u, memory_order_relaxed);

    if (slot < MAX_PER_CELL) {
        cell_list[cell * MAX_PER_CELL + slot] = i;
    }
}

kernel void lj_step(
    device const float4 *pos_in       [[buffer(0)]],
    device       float4 *pos_out      [[buffer(1)]],
    device const float4 *vel_in       [[buffer(2)]],
    device       float4 *vel_out      [[buffer(3)]],
    device const uint   *cell_count   [[buffer(4)]],
    device const uint   *cell_list    [[buffer(5)]],
    constant uint        &N           [[buffer(6)]],
    constant uint        &M           [[buffer(7)]],
    constant float       &L           [[buffer(8)]],
    constant float       &rcut2       [[buffer(9)]],
    constant float       &dt          [[buffer(10)]],
    constant uint        &MAX_PER_CELL[[buffer(11)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= N) return;

    float3 ri = pos_in[i].xyz;
    float3 vi = vel_in[i].xyz;
    float3 a  = float3(0.0f);

    float invL      = 1.0f / L;
    float halfL     = 0.5f * L;
    float inv_cell  = float(M) * invL;
    float cell_size = L / float(M);

    float3 ri_w = lj_wrap_pos(ri, L, invL);

    uint cx = lj_cell_coord(ri_w.x, inv_cell, M);
    uint cy = lj_cell_coord(ri_w.y, inv_cell, M);
    uint cz = lj_cell_coord(ri_w.z, inv_cell, M);

    uint xm = (cx == 0u)       ? (M - 1u) : (cx - 1u);
    uint xp = (cx + 1u == M)   ? 0u       : (cx + 1u);
    uint ym = (cy == 0u)       ? (M - 1u) : (cy - 1u);
    uint yp = (cy + 1u == M)   ? 0u       : (cy + 1u);
    uint zm = (cz == 0u)       ? (M - 1u) : (cz - 1u);
    uint zp = (cz + 1u == M)   ? 0u       : (cz + 1u);

    uint xs[3] = { xm, cx, xp };
    uint ys[3] = { ym, cy, yp };
    uint zs[3] = { zm, cz, zp };

    float fx = ri_w.x - float(cx) * cell_size;
    float fy = ri_w.y - float(cy) * cell_size;
    float fz = ri_w.z - float(cz) * cell_size;

    float dx0 = fx;
    float dx1 = 0.0f;
    float dx2 = cell_size - fx;

    float dy0 = fy;
    float dy1 = 0.0f;
    float dy2 = cell_size - fy;

    float dz0 = fz;
    float dz1 = 0.0f;
    float dz2 = cell_size - fz;

    float dxs2[3] = { dx0 * dx0, dx1, dx2 * dx2 };
    float dys2[3] = { dy0 * dy0, dy1, dy2 * dy2 };
    float dzs2[3] = { dz0 * dz0, dz1, dz2 * dz2 };

    uint MM = M * M;

#pragma unroll
    for (uint iz = 0u; iz < 3u; ++iz) {
        uint zbase = zs[iz] * MM;
        float zmin2 = dzs2[iz];

#pragma unroll
        for (uint iy = 0u; iy < 3u; ++iy) {
            uint yzbase = zbase + ys[iy] * M;
            float yzmin2 = zmin2 + dys2[iy];

#pragma unroll
            for (uint ix = 0u; ix < 3u; ++ix) {
                if (yzmin2 + dxs2[ix] >= rcut2) {
                    continue;
                }

                uint cell = yzbase + xs[ix];
                uint cnt  = cell_count[cell];
                uint base = cell * MAX_PER_CELL;

                for (uint k = 0u; k < cnt; ++k) {
                    uint j = cell_list[base + k];

                    float3 d = pos_in[j].xyz - ri;
                    d = lj_min_image_fast(d, L, halfL);

                    float r2 = dot(d, d);

                    if (r2 < rcut2 && r2 > 1.0e-12f) {
                        float inv_r2 = 1.0f / r2;
                        float inv_r6 = inv_r2 * inv_r2 * inv_r2;
                        float fmag   = 24.0f * inv_r2 * inv_r6 * (1.0f - 2.0f * inv_r6);
                        a = fma(d, float3(fmag), a);
                    }
                }
            }
        }
    }

    float3 v_new = vi + a * dt;
    float3 r_new = ri + v_new * dt;

    pos_out[i] = float4(r_new, 0.0f);
    vel_out[i] = float4(v_new, 0.0f);
}
```