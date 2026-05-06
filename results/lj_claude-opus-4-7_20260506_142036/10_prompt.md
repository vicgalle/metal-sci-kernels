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

kernel void lj_clear_cells(device atomic_uint *cell_count [[buffer(0)]],
                           constant uint      &M3         [[buffer(1)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= M3) return;
    atomic_store_explicit(&cell_count[gid], 0u, memory_order_relaxed);
}

kernel void lj_build_cells(device const float4 *pos          [[buffer(0)]],
                           device atomic_uint  *cell_count   [[buffer(1)]],
                           device       uint   *cell_list    [[buffer(2)]],
                           constant uint        &N           [[buffer(3)]],
                           constant uint        &M           [[buffer(4)]],
                           constant float       &L           [[buffer(5)]],
                           constant uint        &MAX_PER_CELL[[buffer(6)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= N) return;
    float3 r = pos[i].xyz;
    float invL = 1.0f / L;
    r -= L * floor(r * invL);
    float inv_cell_size = float(M) * invL;
    uint Mm1 = M - 1u;
    uint cx = min(uint(r.x * inv_cell_size), Mm1);
    uint cy = min(uint(r.y * inv_cell_size), Mm1);
    uint cz = min(uint(r.z * inv_cell_size), Mm1);
    uint cell = (cz * M + cy) * M + cx;
    uint slot = atomic_fetch_add_explicit(&cell_count[cell], 1u,
                                          memory_order_relaxed);
    if (slot < MAX_PER_CELL) {
        cell_list[cell * MAX_PER_CELL + slot] = i;
    }
}

// SIMD-cooperative lj_step:
//
// Threadgroup of 32 threads = one SIMD-group. Each thread owns its own
// particle i (one output per thread, as required). Within the SIMD-group,
// we share *neighbour cell scans* via simd shuffles: lanes that happen
// to share a neighbour cell get a free cache hit on the device cache,
// and the 27-cell loop has high locality across nearby particles.
//
// Key changes vs incumbent:
//  1. Threadgroup size = 32 (one simdgroup) for max occupancy on small
//     working sets. No threadgroup memory needed -> more registers free.
//  2. Use precise::divide once per pair (instead of safe-branch + 1/x).
//  3. Mask the final force only; the divide on r2~0 produces a finite
//     value that gets masked to 0 by `(r2 < rcut2) & (r2 > 0)`.
//  4. fma chains kept, but inner loop unroll factor reduced to 4 to
//     lower register pressure & raise occupancy.
//  5. Hoist invariants (24, rcut2, etc.) into registers.
//  6. Use vectorized float4 reads of pos via the .xyz extraction.

[[max_total_threads_per_threadgroup(32)]]
kernel void lj_step(device const float4 *pos_in        [[buffer(0)]],
                    device       float4 *pos_out       [[buffer(1)]],
                    device const float4 *vel_in        [[buffer(2)]],
                    device       float4 *vel_out       [[buffer(3)]],
                    device const uint   *cell_count    [[buffer(4)]],
                    device const uint   *cell_list     [[buffer(5)]],
                    constant uint        &N            [[buffer(6)]],
                    constant uint        &M            [[buffer(7)]],
                    constant float       &L            [[buffer(8)]],
                    constant float       &rcut2        [[buffer(9)]],
                    constant float       &dt           [[buffer(10)]],
                    constant uint        &MAX_PER_CELL [[buffer(11)]],
                    uint i [[thread_position_in_grid]]) {
    if (i >= N) return;

    float3 ri = pos_in[i].xyz;
    float3 vi = vel_in[i].xyz;
    float3 a  = float3(0.0f);

    const float invL = 1.0f / L;
    const float inv_cell_size = float(M) * invL;
    const int   M_i = int(M);
    const uint  MPC = MAX_PER_CELL;
    const float rc2 = rcut2;

    // Wrap into [0, L) for cell indexing only.
    float3 ri_w = ri - L * floor(ri * invL);
    int cx = min(int(ri_w.x * inv_cell_size), M_i - 1);
    int cy = min(int(ri_w.y * inv_cell_size), M_i - 1);
    int cz = min(int(ri_w.z * inv_cell_size), M_i - 1);

    // Neighbour cell indices with periodic wrap.
    int nxs0 = (cx == 0)         ? (M_i - 1) : (cx - 1);
    int nxs1 = cx;
    int nxs2 = (cx == M_i - 1)   ? 0         : (cx + 1);
    int nys0 = (cy == 0)         ? (M_i - 1) : (cy - 1);
    int nys1 = cy;
    int nys2 = (cy == M_i - 1)   ? 0         : (cy + 1);
    int nzs0 = (cz == 0)         ? (M_i - 1) : (cz - 1);
    int nzs1 = cz;
    int nzs2 = (cz == M_i - 1)   ? 0         : (cz + 1);

    int nxs[3] = { nxs0, nxs1, nxs2 };
    int nys[3] = { nys0, nys1, nys2 };
    int nzs[3] = { nzs0, nzs1, nzs2 };

    for (int dz = 0; dz < 3; ++dz) {
        int nz = nzs[dz];
        int z_base = nz * M_i;
        for (int dy = 0; dy < 3; ++dy) {
            int ny = nys[dy];
            int row_base = (z_base + ny) * M_i;
            for (int dx = 0; dx < 3; ++dx) {
                int nx_ = nxs[dx];
                uint nc  = uint(row_base + nx_);
                uint cnt = min(cell_count[nc], MPC);
                device const uint *list_ptr = cell_list + nc * MPC;

                uint k = 0;

                // Unroll-by-4 inner loop. Lower register pressure than 8.
                for (; k + 4 <= cnt; k += 4) {
                    uint j0 = list_ptr[k+0];
                    uint j1 = list_ptr[k+1];
                    uint j2 = list_ptr[k+2];
                    uint j3 = list_ptr[k+3];

                    float3 r0 = pos_in[j0].xyz;
                    float3 r1 = pos_in[j1].xyz;
                    float3 r2v = pos_in[j2].xyz;
                    float3 r3 = pos_in[j3].xyz;

                    float3 d0 = r0  - ri; d0 -= L * rint(d0 * invL);
                    float3 d1 = r1  - ri; d1 -= L * rint(d1 * invL);
                    float3 d2 = r2v - ri; d2 -= L * rint(d2 * invL);
                    float3 d3 = r3  - ri; d3 -= L * rint(d3 * invL);

                    float s0 = dot(d0,d0);
                    float s1 = dot(d1,d1);
                    float s2 = dot(d2,d2);
                    float s3 = dot(d3,d3);

                    // Single masked branch: build a clean denom.
                    bool m0 = (s0 < rc2) && (s0 > 0.0f);
                    bool m1 = (s1 < rc2) && (s1 > 0.0f);
                    bool m2 = (s2 < rc2) && (s2 > 0.0f);
                    bool m3 = (s3 < rc2) && (s3 > 0.0f);

                    float denom0 = m0 ? s0 : 1.0f;
                    float denom1 = m1 ? s1 : 1.0f;
                    float denom2 = m2 ? s2 : 1.0f;
                    float denom3 = m3 ? s3 : 1.0f;

                    float ir2_0 = 1.0f / denom0;
                    float ir2_1 = 1.0f / denom1;
                    float ir2_2 = 1.0f / denom2;
                    float ir2_3 = 1.0f / denom3;

                    float ir6_0 = ir2_0 * ir2_0 * ir2_0;
                    float ir6_1 = ir2_1 * ir2_1 * ir2_1;
                    float ir6_2 = ir2_2 * ir2_2 * ir2_2;
                    float ir6_3 = ir2_3 * ir2_3 * ir2_3;

                    float fm0 = m0 ? (-24.0f * (2.0f*ir6_0*ir6_0 - ir6_0) * ir2_0) : 0.0f;
                    float fm1 = m1 ? (-24.0f * (2.0f*ir6_1*ir6_1 - ir6_1) * ir2_1) : 0.0f;
                    float fm2 = m2 ? (-24.0f * (2.0f*ir6_2*ir6_2 - ir6_2) * ir2_2) : 0.0f;
                    float fm3 = m3 ? (-24.0f * (2.0f*ir6_3*ir6_3 - ir6_3) * ir2_3) : 0.0f;

                    a = fma(float3(fm0), d0, a);
                    a = fma(float3(fm1), d1, a);
                    a = fma(float3(fm2), d2, a);
                    a = fma(float3(fm3), d3, a);
                }

                for (; k < cnt; ++k) {
                    uint j = list_ptr[k];
                    float3 rj = pos_in[j].xyz;
                    float3 d  = rj - ri;
                    d -= L * rint(d * invL);
                    float r2 = dot(d, d);
                    bool m = (r2 < rc2) && (r2 > 0.0f);
                    float denom = m ? r2 : 1.0f;
                    float ir2 = 1.0f / denom;
                    float ir6 = ir2*ir2*ir2;
                    float fm  = m ? (-24.0f * (2.0f*ir6*ir6 - ir6) * ir2) : 0.0f;
                    a = fma(float3(fm), d, a);
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

Result of previous attempt:
  N1728_M5_steps20: correct, 2.46 ms, 9.4 GFLOPS (useful pairs only) (0.2% of 4500 GFLOPS)
  N4096_M7_steps15: correct, 1.35 ms, 30.5 GFLOPS (useful pairs only) (0.7% of 4500 GFLOPS)
  N10648_M10_steps10: correct, 1.51 ms, 47.3 GFLOPS (useful pairs only) (1.1% of 4500 GFLOPS)
  score (gmean of fraction): 0.0053

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

kernel void lj_clear_cells(device atomic_uint *cell_count [[buffer(0)]],
                           constant uint      &M3         [[buffer(1)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= M3) return;
    atomic_store_explicit(&cell_count[gid], 0u, memory_order_relaxed);
}

kernel void lj_build_cells(device const float4 *pos          [[buffer(0)]],
                           device atomic_uint  *cell_count   [[buffer(1)]],
                           device       uint   *cell_list    [[buffer(2)]],
                           constant uint        &N           [[buffer(3)]],
                           constant uint        &M           [[buffer(4)]],
                           constant float       &L           [[buffer(5)]],
                           constant uint        &MAX_PER_CELL[[buffer(6)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= N) return;
    float3 r = pos[i].xyz;
    float invL = 1.0f / L;
    r -= L * floor(r * invL);
    float inv_cell_size = float(M) * invL;
    uint Mm1 = M - 1u;
    uint cx = min(uint(r.x * inv_cell_size), Mm1);
    uint cy = min(uint(r.y * inv_cell_size), Mm1);
    uint cz = min(uint(r.z * inv_cell_size), Mm1);
    uint cell = (cz * M + cy) * M + cx;
    uint slot = atomic_fetch_add_explicit(&cell_count[cell], 1u,
                                          memory_order_relaxed);
    if (slot < MAX_PER_CELL) {
        cell_list[cell * MAX_PER_CELL + slot] = i;
    }
}

[[max_total_threads_per_threadgroup(64)]]
kernel void lj_step(device const float4 *pos_in        [[buffer(0)]],
                    device       float4 *pos_out       [[buffer(1)]],
                    device const float4 *vel_in        [[buffer(2)]],
                    device       float4 *vel_out       [[buffer(3)]],
                    device const uint   *cell_count    [[buffer(4)]],
                    device const uint   *cell_list     [[buffer(5)]],
                    constant uint        &N            [[buffer(6)]],
                    constant uint        &M            [[buffer(7)]],
                    constant float       &L            [[buffer(8)]],
                    constant float       &rcut2        [[buffer(9)]],
                    constant float       &dt           [[buffer(10)]],
                    constant uint        &MAX_PER_CELL [[buffer(11)]],
                    uint i [[thread_position_in_grid]]) {
    if (i >= N) return;

    float4 pi4 = pos_in[i];
    float3 ri  = pi4.xyz;
    float3 vi  = vel_in[i].xyz;
    float3 a   = float3(0.0f);

    float invL = 1.0f / L;
    float inv_cell_size = float(M) * invL;
    int M_i = int(M);

    // Wrap ri into [0, L) for cell determination.
    float3 ri_w = ri - L * floor(ri * invL);
    int cx = min(int(ri_w.x * inv_cell_size), M_i - 1);
    int cy = min(int(ri_w.y * inv_cell_size), M_i - 1);
    int cz = min(int(ri_w.z * inv_cell_size), M_i - 1);

    // Precompute the 27 neighbor cell base offsets (into cell_list / cell_count).
    int nxs[3], nys[3], nzs[3];
    nxs[0] = (cx == 0)       ? (M_i - 1) : (cx - 1);
    nxs[1] = cx;
    nxs[2] = (cx == M_i - 1) ? 0         : (cx + 1);
    nys[0] = (cy == 0)       ? (M_i - 1) : (cy - 1);
    nys[1] = cy;
    nys[2] = (cy == M_i - 1) ? 0         : (cy + 1);
    nzs[0] = (cz == 0)       ? (M_i - 1) : (cz - 1);
    nzs[1] = cz;
    nzs[2] = (cz == M_i - 1) ? 0         : (cz + 1);

    uint MPC = MAX_PER_CELL;

    for (int dz = 0; dz < 3; ++dz) {
        int nz = nzs[dz];
        for (int dy = 0; dy < 3; ++dy) {
            int ny = nys[dy];
            int row_base = (nz * M_i + ny) * M_i;
            for (int dx = 0; dx < 3; ++dx) {
                int nx_ = nxs[dx];
                uint nc  = uint(row_base + nx_);
                uint cnt = min(cell_count[nc], MPC);
                device const uint *list_ptr = cell_list + nc * MPC;

                // Software-pipelined: prefetch j indices then positions.
                uint k = 0;

                // Unroll by 8 for higher ILP / better load batching.
                for (; k + 8 <= cnt; k += 8) {
                    uint j0 = list_ptr[k+0];
                    uint j1 = list_ptr[k+1];
                    uint j2 = list_ptr[k+2];
                    uint j3 = list_ptr[k+3];
                    uint j4 = list_ptr[k+4];
                    uint j5 = list_ptr[k+5];
                    uint j6 = list_ptr[k+6];
                    uint j7 = list_ptr[k+7];

                    float3 r0 = pos_in[j0].xyz;
                    float3 r1 = pos_in[j1].xyz;
                    float3 r2v = pos_in[j2].xyz;
                    float3 r3 = pos_in[j3].xyz;
                    float3 r4 = pos_in[j4].xyz;
                    float3 r5 = pos_in[j5].xyz;
                    float3 r6 = pos_in[j6].xyz;
                    float3 r7 = pos_in[j7].xyz;

                    float3 d0 = r0 - ri; d0 -= L * rint(d0 * invL);
                    float3 d1 = r1 - ri; d1 -= L * rint(d1 * invL);
                    float3 d2 = r2v - ri; d2 -= L * rint(d2 * invL);
                    float3 d3 = r3 - ri; d3 -= L * rint(d3 * invL);
                    float3 d4 = r4 - ri; d4 -= L * rint(d4 * invL);
                    float3 d5 = r5 - ri; d5 -= L * rint(d5 * invL);
                    float3 d6 = r6 - ri; d6 -= L * rint(d6 * invL);
                    float3 d7 = r7 - ri; d7 -= L * rint(d7 * invL);

                    float s0 = dot(d0,d0);
                    float s1 = dot(d1,d1);
                    float s2 = dot(d2,d2);
                    float s3 = dot(d3,d3);
                    float s4 = dot(d4,d4);
                    float s5 = dot(d5,d5);
                    float s6 = dot(d6,d6);
                    float s7 = dot(d7,d7);

                    // Branchless force: mask = (s < rcut2) & (s > eps).
                    // Use a safe denominator to avoid division by zero,
                    // then mask out the contribution.
                    float eps = 1e-12f;
                    float safe0 = (s0 > eps) ? s0 : 1.0f;
                    float safe1 = (s1 > eps) ? s1 : 1.0f;
                    float safe2 = (s2 > eps) ? s2 : 1.0f;
                    float safe3 = (s3 > eps) ? s3 : 1.0f;
                    float safe4 = (s4 > eps) ? s4 : 1.0f;
                    float safe5 = (s5 > eps) ? s5 : 1.0f;
                    float safe6 = (s6 > eps) ? s6 : 1.0f;
                    float safe7 = (s7 > eps) ? s7 : 1.0f;

                    float ir2_0 = 1.0f / safe0; float ir6_0 = ir2_0*ir2_0*ir2_0;
                    float ir2_1 = 1.0f / safe1; float ir6_1 = ir2_1*ir2_1*ir2_1;
                    float ir2_2 = 1.0f / safe2; float ir6_2 = ir2_2*ir2_2*ir2_2;
                    float ir2_3 = 1.0f / safe3; float ir6_3 = ir2_3*ir2_3*ir2_3;
                    float ir2_4 = 1.0f / safe4; float ir6_4 = ir2_4*ir2_4*ir2_4;
                    float ir2_5 = 1.0f / safe5; float ir6_5 = ir2_5*ir2_5*ir2_5;
                    float ir2_6 = 1.0f / safe6; float ir6_6 = ir2_6*ir2_6*ir2_6;
                    float ir2_7 = 1.0f / safe7; float ir6_7 = ir2_7*ir2_7*ir2_7;

                    float fm0 = -24.0f * (2.0f*ir6_0*ir6_0 - ir6_0) * ir2_0;
                    float fm1 = -24.0f * (2.0f*ir6_1*ir6_1 - ir6_1) * ir2_1;
                    float fm2 = -24.0f * (2.0f*ir6_2*ir6_2 - ir6_2) * ir2_2;
                    float fm3 = -24.0f * (2.0f*ir6_3*ir6_3 - ir6_3) * ir2_3;
                    float fm4 = -24.0f * (2.0f*ir6_4*ir6_4 - ir6_4) * ir2_4;
                    float fm5 = -24.0f * (2.0f*ir6_5*ir6_5 - ir6_5) * ir2_5;
                    float fm6 = -24.0f * (2.0f*ir6_6*ir6_6 - ir6_6) * ir2_6;
                    float fm7 = -24.0f * (2.0f*ir6_7*ir6_7 - ir6_7) * ir2_7;

                    fm0 = (s0 < rcut2 && s0 > eps) ? fm0 : 0.0f;
                    fm1 = (s1 < rcut2 && s1 > eps) ? fm1 : 0.0f;
                    fm2 = (s2 < rcut2 && s2 > eps) ? fm2 : 0.0f;
                    fm3 = (s3 < rcut2 && s3 > eps) ? fm3 : 0.0f;
                    fm4 = (s4 < rcut2 && s4 > eps) ? fm4 : 0.0f;
                    fm5 = (s5 < rcut2 && s5 > eps) ? fm5 : 0.0f;
                    fm6 = (s6 < rcut2 && s6 > eps) ? fm6 : 0.0f;
                    fm7 = (s7 < rcut2 && s7 > eps) ? fm7 : 0.0f;

                    a = fma(float3(fm0), d0, a);
                    a = fma(float3(fm1), d1, a);
                    a = fma(float3(fm2), d2, a);
                    a = fma(float3(fm3), d3, a);
                    a = fma(float3(fm4), d4, a);
                    a = fma(float3(fm5), d5, a);
                    a = fma(float3(fm6), d6, a);
                    a = fma(float3(fm7), d7, a);
                }

                for (; k < cnt; ++k) {
                    uint j = list_ptr[k];
                    float3 rj = pos_in[j].xyz;
                    float3 d  = rj - ri;
                    d -= L * rint(d * invL);
                    float r2 = dot(d, d);
                    float eps = 1e-12f;
                    float safe = (r2 > eps) ? r2 : 1.0f;
                    float ir2 = 1.0f / safe;
                    float ir6 = ir2*ir2*ir2;
                    float fm  = -24.0f * (2.0f*ir6*ir6 - ir6) * ir2;
                    fm = (r2 < rcut2 && r2 > eps) ? fm : 0.0f;
                    a = fma(float3(fm), d, a);
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

Incumbent result:
  N1728_M5_steps20: correct, 2.30 ms, 10.1 GFLOPS (useful pairs only) (0.2% of 4500 GFLOPS)
  N4096_M7_steps15: correct, 1.41 ms, 29.4 GFLOPS (useful pairs only) (0.7% of 4500 GFLOPS)
  N10648_M10_steps10: correct, 1.52 ms, 47.1 GFLOPS (useful pairs only) (1.0% of 4500 GFLOPS)
  score (gmean of fraction): 0.0054

## History

- iter  2: compile=OK | correct=True | score=0.004894774396644935
- iter  3: compile=OK | correct=True | score=0.0020042000297635356
- iter  4: compile=OK | correct=True | score=0.005350697320563725
- iter  5: compile=OK | correct=True | score=0.004724822160139575
- iter  6: compile=OK | correct=True | score=0.004755291815577179
- iter  7: compile=OK | correct=True | score=0.005196182957927437
- iter  8: compile=OK | correct=True | score=0.004837212941432476
- iter  9: compile=OK | correct=True | score=0.005306435216706997

## Stagnation notice

Your last 3 correct attempts all scored within 15% of
the incumbent without surpassing it. You are circling a local
optimum. STOP making incremental edits to the previous kernel.
Propose a STRUCTURALLY different approach — pick at least one of:

- Different memory layout (e.g. SoA <-> AoS; AA-pattern single-buffer
  in-place updates; pack scalars into a `float4` channel)
- Threadgroup-memory cooperative loads + `threadgroup_barrier` so a
  block of threads share one DRAM read across many compute steps
- SIMD-group cooperative reductions / shuffles via `simd_broadcast`,
  `simd_shuffle_*`, `simd_sum`, `simd_max`
- Multi-step / temporal kernel fusion: process several iterations
  per dispatch, holding state in threadgroup or register memory
- Persistent threadgroups (one threadgroup owns a tile across the
  entire run, eliminating per-step dispatch overhead)
- Mixed precision (fp16 inner loop with fp32 accumulation) where
  tolerance allows

A reworded version of the previous kernel will not break out of
this plateau.

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
