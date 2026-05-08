## Task: lj

Lennard-Jones molecular dynamics with a cell-list spatial hash. Cubic periodic box of side L; cutoff rcut = 2.5 (sigma = epsilon = mass = 1).

Per timestep, three kernels are dispatched in this fixed order:
  1) lj_clear_cells: zero the per-cell occupancy counter (M^3 threads).
  2) lj_build_cells: each particle thread computes its cell index (after wrapping its position into [0, L)) and atomically appends itself to that cell (N threads).
  3) lj_step: each particle thread iterates the 27 neighbour cells (its own cell + 3^3 - 1 face/edge/corner neighbours, with periodic wrap), reads each occupant from cell_list, and sums the Lennard-Jones force from those within rcut. It then takes one symplectic-Euler step:  v_new = v + a*dt; r_new = r + v_new*dt (N threads).

Cell layout: M cells per side; cell index = (cz*M + cy)*M + cx; cell_size = L/M is guaranteed >= rcut so 27 neighbour cells cover all interactions. cell_count[M^3] holds the per-cell occupancy, cell_list[M^3 * MAX_PER_CELL] holds the particle indices, with row-major slot order. MAX_PER_CELL = 64 is generous for the supplied initial states; particles exceeding this cap are silently dropped (the seed tolerates this since the well-conditioned initial state never overflows, and a candidate may rely on the same invariant).

Lennard-Jones force on i from j (sigma = epsilon = 1):
  d = (r_j - r_i), minimum-image:  d -= L * round(d / L)
  r2 = dot(d, d); skip if r2 >= rcut^2 or r2 ~= 0
  inv_r2 = 1/r2; inv_r6 = inv_r2^3; inv_r12 = inv_r6^2
  F_on_i = -24 * (2*inv_r12 - inv_r6) * inv_r2 * d
  a_i = sum of F_on_i over all j within cutoff (mass = 1).

Positions/velocities are stored as float4 with .xyz holding the data and .w padding (matches the nbody task's layout). The host ping-pongs (pos_in, pos_out) and (vel_in, vel_out) buffer pairs each step; cell_count and cell_list are scratch buffers reused every step (cleared by lj_clear_cells).

## Required kernel signature(s)

```
kernel void lj_clear_cells(
    device atomic_uint *cell_count [[buffer(0)]],
    constant uint      &M3         [[buffer(1)]],
    uint gid [[thread_position_in_grid]]);

kernel void lj_build_cells(
    device const float4 *pos          [[buffer(0)]],
    device atomic_uint  *cell_count   [[buffer(1)]],
    device       uint   *cell_list    [[buffer(2)]],
    constant uint        &N           [[buffer(3)]],
    constant uint        &M           [[buffer(4)]],
    constant float       &L           [[buffer(5)]],
    constant uint        &MAX_PER_CELL[[buffer(6)]],
    uint i [[thread_position_in_grid]]);

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
    uint i [[thread_position_in_grid]]);

All three kernels are dispatched 1-D, one thread per element. lj_clear_cells: M^3 threads (gid >= M3 early-exits). lj_build_cells / lj_step: N threads (i >= N early-exits). Each thread MUST handle exactly one element; the host will not shrink the dispatch if you process multiple elements per thread. All buffers use MTLResourceStorageModeShared (Apple Silicon unified memory). cell_count is read via atomics from lj_clear_cells / lj_build_cells and as a plain uint* in lj_step (no atomicity required for the read-only pass).
```

## Your previous attempt

```metal
#include <metal_stdlib>
using namespace metal;

#define LJ_MAX_PER_CELL_FIXED 64u
#define LJ_CELL_SHIFT 6u

static inline float lj_wrap_scalar_fast(float x, float L, float invL) {
    if (x >= L) {
        x -= L;
        if (x >= L) {
            x -= L * floor(x * invL);
        }
    } else if (x < 0.0f) {
        x += L;
        if (x < 0.0f) {
            x -= L * floor(x * invL);
        }
    }
    return x;
}

static inline float3 lj_wrap_pos_fast(float3 r, float L, float invL) {
    return float3(lj_wrap_scalar_fast(r.x, L, invL),
                  lj_wrap_scalar_fast(r.y, L, invL),
                  lj_wrap_scalar_fast(r.z, L, invL));
}

static inline uint lj_cell_coord(float x, float inv_cell, uint M) {
    return min(uint(x * inv_cell), M - 1u);
}

static inline float3 lj_accum_cell64(
    device const float4 *pos_in,
    device const uint   *cell_count,
    device const uint   *cell_list,
    uint cell,
    uint self_i,
    float rix,
    float riy,
    float riz,
    float L,
    float halfL,
    float negHalfL,
    float rcut2,
    float3 a)
{
    uint cnt = cell_count[cell];
    cnt = min(cnt, LJ_MAX_PER_CELL_FIXED);
    uint base = cell << LJ_CELL_SHIFT;

#define LJ_ACCUM_PAIR(PJ4, JIDX)                                             \
    do {                                                                     \
        float dx = (PJ4).x - rix;                                            \
        float dy = (PJ4).y - riy;                                            \
        float dz = (PJ4).z - riz;                                            \
                                                                             \
        if (dx > halfL) {                                                    \
            dx -= L;                                                         \
        } else if (dx < negHalfL) {                                          \
            dx += L;                                                         \
        }                                                                    \
                                                                             \
        if (dy > halfL) {                                                    \
            dy -= L;                                                         \
        } else if (dy < negHalfL) {                                          \
            dy += L;                                                         \
        }                                                                    \
                                                                             \
        if (dz > halfL) {                                                    \
            dz -= L;                                                         \
        } else if (dz < negHalfL) {                                          \
            dz += L;                                                         \
        }                                                                    \
                                                                             \
        float r2 = fma(dx, dx, fma(dy, dy, dz * dz));                        \
                                                                             \
        if (r2 < rcut2 && (JIDX) != self_i) {                                \
            float inv_r2 = 1.0f / r2;                                        \
            float inv_r6 = inv_r2 * inv_r2 * inv_r2;                         \
            float fmag   = 24.0f * inv_r2 * inv_r6 * (1.0f - 2.0f * inv_r6); \
                                                                             \
            a.x = fma(fmag, dx, a.x);                                        \
            a.y = fma(fmag, dy, a.y);                                        \
            a.z = fma(fmag, dz, a.z);                                        \
        }                                                                    \
    } while (0)

    uint n2 = cnt & ~1u;
    for (uint k = 0u; k < n2; k += 2u) {
        uint j0 = cell_list[base + k];
        uint j1 = cell_list[base + k + 1u];

        float4 pj0 = pos_in[j0];
        float4 pj1 = pos_in[j1];

        LJ_ACCUM_PAIR(pj0, j0);
        LJ_ACCUM_PAIR(pj1, j1);
    }

    if ((cnt & 1u) != 0u) {
        uint j = cell_list[base + n2];
        float4 pj = pos_in[j];
        LJ_ACCUM_PAIR(pj, j);
    }

#undef LJ_ACCUM_PAIR

    return a;
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

    float3 r = lj_wrap_pos_fast(pos[i].xyz, L, invL);

    uint cx = lj_cell_coord(r.x, inv_cell, M);
    uint cy = lj_cell_coord(r.y, inv_cell, M);
    uint cz = lj_cell_coord(r.z, inv_cell, M);

    uint cell = (cz * M + cy) * M + cx;
    uint slot = atomic_fetch_add_explicit(&cell_count[cell], 1u, memory_order_relaxed);

    if (slot < LJ_MAX_PER_CELL_FIXED) {
        cell_list[(cell << LJ_CELL_SHIFT) + slot] = i;
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

    float4 pi = pos_in[i];

    float rix = pi.x;
    float riy = pi.y;
    float riz = pi.z;

    float invL      = 1.0f / L;
    float halfL     = 0.5f * L;
    float negHalfL  = -halfL;
    float inv_cell  = float(M) * invL;
    float cell_size = L / float(M);

    float3 ri_w = lj_wrap_pos_fast(float3(rix, riy, riz), L, invL);

    uint cx = lj_cell_coord(ri_w.x, inv_cell, M);
    uint cy = lj_cell_coord(ri_w.y, inv_cell, M);
    uint cz = lj_cell_coord(ri_w.z, inv_cell, M);

    uint xm = (cx == 0u)     ? (M - 1u) : (cx - 1u);
    uint xp = (cx + 1u == M) ? 0u       : (cx + 1u);
    uint ym = (cy == 0u)     ? (M - 1u) : (cy - 1u);
    uint yp = (cy + 1u == M) ? 0u       : (cy + 1u);
    uint zm = (cz == 0u)     ? (M - 1u) : (cz - 1u);
    uint zp = (cz + 1u == M) ? 0u       : (cz + 1u);

    float fx = ri_w.x - float(cx) * cell_size;
    float fy = ri_w.y - float(cy) * cell_size;
    float fz = ri_w.z - float(cz) * cell_size;

    float dxm2 = fx * fx;
    float dxp  = cell_size - fx;
    float dxp2 = dxp * dxp;

    float dym2 = fy * fy;
    float dyp  = cell_size - fy;
    float dyp2 = dyp * dyp;

    float dzm2 = fz * fz;
    float dzp  = cell_size - fz;
    float dzp2 = dzp * dzp;

    uint MM  = M * M;
    uint x0  = xm;
    uint x1  = cx;
    uint x2  = xp;
    uint y0M = ym * M;
    uint y1M = cy * M;
    uint y2M = yp * M;
    uint z0B = zm * MM;
    uint z1B = cz * MM;
    uint z2B = zp * MM;

    float3 a = float3(0.0f);

#define LJ_TRY_CELL(CELL_EXPR, D2_EXPR)                                      \
    do {                                                                     \
        if ((D2_EXPR) < rcut2) {                                             \
            a = lj_accum_cell64(pos_in, cell_count, cell_list,               \
                                (CELL_EXPR), i, rix, riy, riz,              \
                                L, halfL, negHalfL, rcut2, a);               \
        }                                                                    \
    } while (0)

    float zy;

    zy = dzm2 + dym2;
    LJ_TRY_CELL(z0B + y0M + x0, zy + dxm2);
    LJ_TRY_CELL(z0B + y0M + x1, zy);
    LJ_TRY_CELL(z0B + y0M + x2, zy + dxp2);

    zy = dzm2;
    LJ_TRY_CELL(z0B + y1M + x0, zy + dxm2);
    LJ_TRY_CELL(z0B + y1M + x1, zy);
    LJ_TRY_CELL(z0B + y1M + x2, zy + dxp2);

    zy = dzm2 + dyp2;
    LJ_TRY_CELL(z0B + y2M + x0, zy + dxm2);
    LJ_TRY_CELL(z0B + y2M + x1, zy);
    LJ_TRY_CELL(z0B + y2M + x2, zy + dxp2);

    zy = dym2;
    LJ_TRY_CELL(z1B + y0M + x0, zy + dxm2);
    LJ_TRY_CELL(z1B + y0M + x1, zy);
    LJ_TRY_CELL(z1B + y0M + x2, zy + dxp2);

    LJ_TRY_CELL(z1B + y1M + x0, dxm2);
    LJ_TRY_CELL(z1B + y1M + x1, 0.0f);
    LJ_TRY_CELL(z1B + y1M + x2, dxp2);

    zy = dyp2;
    LJ_TRY_CELL(z1B + y2M + x0, zy + dxm2);
    LJ_TRY_CELL(z1B + y2M + x1, zy);
    LJ_TRY_CELL(z1B + y2M + x2, zy + dxp2);

    zy = dzp2 + dym2;
    LJ_TRY_CELL(z2B + y0M + x0, zy + dxm2);
    LJ_TRY_CELL(z2B + y0M + x1, zy);
    LJ_TRY_CELL(z2B + y0M + x2, zy + dxp2);

    zy = dzp2;
    LJ_TRY_CELL(z2B + y1M + x0, zy + dxm2);
    LJ_TRY_CELL(z2B + y1M + x1, zy);
    LJ_TRY_CELL(z2B + y1M + x2, zy + dxp2);

    zy = dzp2 + dyp2;
    LJ_TRY_CELL(z2B + y2M + x0, zy + dxm2);
    LJ_TRY_CELL(z2B + y2M + x1, zy);
    LJ_TRY_CELL(z2B + y2M + x2, zy + dxp2);

#undef LJ_TRY_CELL

    float4 vi4 = vel_in[i];
    float3 v_new = vi4.xyz + a * dt;
    float3 r_new = float3(rix, riy, riz) + v_new * dt;

    pos_out[i] = float4(r_new, 0.0f);
    vel_out[i] = float4(v_new, 0.0f);
}
```

Result of previous attempt:
  N1728_M5_steps20: correct, 2.39 ms, 9.7 GFLOPS (useful pairs only) (0.2% of 4500 GFLOPS)
  N4096_M7_steps15: correct, 2.08 ms, 19.9 GFLOPS (useful pairs only) (0.4% of 4500 GFLOPS)
  N10648_M10_steps10: correct, 1.71 ms, 41.7 GFLOPS (useful pairs only) (0.9% of 4500 GFLOPS)
  score (gmean of fraction): 0.0045

## History

- iter  0: compile=OK | correct=True | score=0.00310240244644033
- iter  1: compile=OK | correct=True | score=0.0031453404444332514
- iter  2: compile=OK | correct=True | score=0.003193822967484982
- iter  3: compile=OK | correct=True | score=0.003291818437960419
- iter  4: compile=OK | correct=True | score=0.002886435556640643
- iter  5: compile=OK | correct=True | score=0.00445710940463453

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
