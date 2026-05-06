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

#define LJ_TG 64
#define LJ_MAX_SHARED 1792  // 27 * 64 + a bit; safely covers MPC<=64

[[max_total_threads_per_threadgroup(LJ_TG)]]
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
                    uint i      [[thread_position_in_grid]],
                    uint tid    [[thread_index_in_threadgroup]],
                    uint tgid   [[threadgroup_position_in_grid]]) {
    threadgroup float3 sh_pos[LJ_MAX_SHARED];
    threadgroup uint   sh_count;
    threadgroup int    sh_anchor_cx;
    threadgroup int    sh_anchor_cy;
    threadgroup int    sh_anchor_cz;

    bool active = (i < N);

    float4 pi4 = active ? pos_in[i] : float4(0.0f);
    float3 ri  = pi4.xyz;
    float3 vi  = active ? vel_in[i].xyz : float3(0.0f);

    float invL = 1.0f / L;
    float Mf   = float(M);
    float inv_cell_size = Mf * invL;
    int M_i = int(M);
    uint MPC = MAX_PER_CELL;

    // Anchor cell: chosen as the cell of thread 0's particle in this group.
    // Threadgroup index 0's particle index is tgid * LJ_TG.
    if (tid == 0) {
        uint anchor_idx = min(tgid * LJ_TG, N - 1u);
        float3 ra = pos_in[anchor_idx].xyz;
        ra -= L * floor(ra * invL);
        int acx = min(int(ra.x * inv_cell_size), M_i - 1);
        int acy = min(int(ra.y * inv_cell_size), M_i - 1);
        int acz = min(int(ra.z * inv_cell_size), M_i - 1);
        sh_anchor_cx = acx;
        sh_anchor_cy = acy;
        sh_anchor_cz = acz;
        sh_count = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int acx = sh_anchor_cx;
    int acy = sh_anchor_cy;
    int acz = sh_anchor_cz;

    // Cooperative load of 27 anchor-neighbour cells into threadgroup memory.
    // Thread tid handles neighbour cell index tid (0..26) if tid<27.
    if (tid < 27u) {
        int dz = int(tid / 9u) - 1;
        int rem = int(tid % 9u);
        int dy = rem / 3 - 1;
        int dx = rem % 3 - 1;
        int nx = acx + dx; if (nx < 0) nx += M_i; else if (nx >= M_i) nx -= M_i;
        int ny = acy + dy; if (ny < 0) ny += M_i; else if (ny >= M_i) ny -= M_i;
        int nz = acz + dz; if (nz < 0) nz += M_i; else if (nz >= M_i) nz -= M_i;
        uint nc  = uint((nz * M_i + ny) * M_i + nx);
        uint cnt = min(cell_count[nc], MPC);
        // Reserve slots in shared pool.
        uint base;
        // Use atomic on threadgroup memory? Simpler: serialize via one writer per neighbour
        // by atomically incrementing sh_count (cast).
        threadgroup atomic_uint *sh_count_a = (threadgroup atomic_uint *)&sh_count;
        base = atomic_fetch_add_explicit(sh_count_a, cnt, memory_order_relaxed);
        device const uint *lp = cell_list + nc * MPC;
        for (uint k = 0; k < cnt; ++k) {
            uint j = lp[k];
            if (base + k < LJ_MAX_SHARED) {
                sh_pos[base + k] = pos_in[j].xyz;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint total = min(sh_count, (uint)LJ_MAX_SHARED);

    // Determine if this thread's particle is actually in the anchor's 27-cell shell.
    // For correctness, fall back to the slow path if not.
    bool in_anchor_shell = false;
    if (active) {
        float3 ri_w = ri - L * floor(ri * invL);
        int cx = min(int(ri_w.x * inv_cell_size), M_i - 1);
        int cy = min(int(ri_w.y * inv_cell_size), M_i - 1);
        int cz = min(int(ri_w.z * inv_cell_size), M_i - 1);
        // Periodic distance in cell coords.
        int ddx = cx - acx; if (ddx >  M_i/2) ddx -= M_i; else if (ddx < -M_i/2) ddx += M_i;
        int ddy = cy - acy; if (ddy >  M_i/2) ddy -= M_i; else if (ddy < -M_i/2) ddy += M_i;
        int ddz = cz - acz; if (ddz >  M_i/2) ddz -= M_i; else if (ddz < -M_i/2) ddz += M_i;
        // To use shared pool (which covers neighbours of anchor = anchor +/- 1),
        // particle's own cell must be within 1 of anchor in every axis (so its
        // neighbours are within 2 of anchor — NOT covered). We need particle's
        // own neighbours all be inside the loaded set, i.e. particle's cell ==
        // anchor cell (so its 27 neighbours == anchor's 27 neighbours).
        in_anchor_shell = (ddx == 0 && ddy == 0 && ddz == 0);
    }

    float3 a = float3(0.0f);

    if (in_anchor_shell) {
        // Fast path: iterate shared pool.
        for (uint k = 0; k < total; ++k) {
            float3 rj = sh_pos[k];
            float3 d  = rj - ri;
            d -= L * rint(d * invL);
            float r2 = dot(d, d);
            if (r2 < rcut2 && r2 > 1e-12f) {
                float ir2 = 1.0f / r2;
                float ir6 = ir2 * ir2 * ir2;
                float fm  = -24.0f * (2.0f * ir6 * ir6 - ir6) * ir2;
                a = fma(float3(fm), d, a);
            }
        }
    } else if (active) {
        // Slow path: original cell-walking logic.
        float3 ri_w = ri - L * floor(ri * invL);
        int cx = min(int(ri_w.x * inv_cell_size), M_i - 1);
        int cy = min(int(ri_w.y * inv_cell_size), M_i - 1);
        int cz = min(int(ri_w.z * inv_cell_size), M_i - 1);
        int nxs[3], nys[3], nzs[3];
        nxs[0] = (cx == 0) ? (M_i - 1) : (cx - 1);
        nxs[1] = cx;
        nxs[2] = (cx == M_i - 1) ? 0 : (cx + 1);
        nys[0] = (cy == 0) ? (M_i - 1) : (cy - 1);
        nys[1] = cy;
        nys[2] = (cy == M_i - 1) ? 0 : (cy + 1);
        nzs[0] = (cz == 0) ? (M_i - 1) : (cz - 1);
        nzs[1] = cz;
        nzs[2] = (cz == M_i - 1) ? 0 : (cz + 1);
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
                    for (uint k = 0; k < cnt; ++k) {
                        uint j = list_ptr[k];
                        float3 rj = pos_in[j].xyz;
                        float3 d  = rj - ri;
                        d -= L * rint(d * invL);
                        float r2 = dot(d, d);
                        if (r2 < rcut2 && r2 > 1e-12f) {
                            float ir2 = 1.0f / r2;
                            float ir6 = ir2 * ir2 * ir2;
                            float fm  = -24.0f * (2.0f * ir6 * ir6 - ir6) * ir2;
                            a = fma(float3(fm), d, a);
                        }
                    }
                }
            }
        }
    }

    if (active) {
        float3 v_new = vi + a * dt;
        float3 r_new = ri + v_new * dt;
        pos_out[i] = float4(r_new, 0.0f);
        vel_out[i] = float4(v_new, 0.0f);
    }
}
```

Result of previous attempt:
  N1728_M5_steps20: correct, 3.20 ms, 7.3 GFLOPS (useful pairs only) (0.2% of 4500 GFLOPS)
  N4096_M7_steps15: correct, 4.17 ms, 9.9 GFLOPS (useful pairs only) (0.2% of 4500 GFLOPS)
  N10648_M10_steps10: correct, 7.02 ms, 10.2 GFLOPS (useful pairs only) (0.2% of 4500 GFLOPS)
  score (gmean of fraction): 0.0020

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
    bool active = (i < N);

    float4 pi4 = active ? pos_in[i] : float4(0.0f);
    float3 ri  = pi4.xyz;
    float3 vi  = active ? vel_in[i].xyz : float3(0.0f);
    float3 a   = float3(0.0f);

    float invL = 1.0f / L;
    float Mf   = float(M);
    float inv_cell_size = Mf * invL;

    // Wrap ri into [0, L) for cell determination only.
    float3 ri_w = ri - L * floor(ri * invL);
    int M_i = int(M);
    int cx = min(int(ri_w.x * inv_cell_size), M_i - 1);
    int cy = min(int(ri_w.y * inv_cell_size), M_i - 1);
    int cz = min(int(ri_w.z * inv_cell_size), M_i - 1);

    // Precompute wrapped neighbor coordinate arrays (branchless).
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

                // Unroll by 4 to expose ILP and hide load latency.
                uint k = 0;
                for (; k + 4 <= cnt; k += 4) {
                    uint j0 = list_ptr[k+0];
                    uint j1 = list_ptr[k+1];
                    uint j2 = list_ptr[k+2];
                    uint j3 = list_ptr[k+3];
                    float3 r0 = pos_in[j0].xyz;
                    float3 r1 = pos_in[j1].xyz;
                    float3 r2v = pos_in[j2].xyz;
                    float3 r3 = pos_in[j3].xyz;

                    float3 d0 = r0 - ri; d0 -= L * rint(d0 * invL);
                    float3 d1 = r1 - ri; d1 -= L * rint(d1 * invL);
                    float3 d2 = r2v - ri; d2 -= L * rint(d2 * invL);
                    float3 d3 = r3 - ri; d3 -= L * rint(d3 * invL);

                    float s0 = dot(d0,d0);
                    float s1 = dot(d1,d1);
                    float s2 = dot(d2,d2);
                    float s3 = dot(d3,d3);

                    if (s0 < rcut2 && s0 > 1e-12f) {
                        float ir2 = 1.0f / s0;
                        float ir6 = ir2*ir2*ir2;
                        float fm = -24.0f * (2.0f*ir6*ir6 - ir6) * ir2;
                        a = fma(float3(fm), d0, a);
                    }
                    if (s1 < rcut2 && s1 > 1e-12f) {
                        float ir2 = 1.0f / s1;
                        float ir6 = ir2*ir2*ir2;
                        float fm = -24.0f * (2.0f*ir6*ir6 - ir6) * ir2;
                        a = fma(float3(fm), d1, a);
                    }
                    if (s2 < rcut2 && s2 > 1e-12f) {
                        float ir2 = 1.0f / s2;
                        float ir6 = ir2*ir2*ir2;
                        float fm = -24.0f * (2.0f*ir6*ir6 - ir6) * ir2;
                        a = fma(float3(fm), d2, a);
                    }
                    if (s3 < rcut2 && s3 > 1e-12f) {
                        float ir2 = 1.0f / s3;
                        float ir6 = ir2*ir2*ir2;
                        float fm = -24.0f * (2.0f*ir6*ir6 - ir6) * ir2;
                        a = fma(float3(fm), d3, a);
                    }
                }
                for (; k < cnt; ++k) {
                    uint j = list_ptr[k];
                    float3 rj = pos_in[j].xyz;
                    float3 d  = rj - ri;
                    d -= L * rint(d * invL);
                    float r2 = dot(d, d);
                    if (r2 < rcut2 && r2 > 1e-12f) {
                        float inv_r2  = 1.0f / r2;
                        float inv_r6  = inv_r2 * inv_r2 * inv_r2;
                        float inv_r12 = inv_r6 * inv_r6;
                        float fmag = -24.0f * (2.0f * inv_r12 - inv_r6) * inv_r2;
                        a = fma(float3(fmag), d, a);
                    }
                }
            }
        }
    }

    if (active) {
        float3 v_new = vi + a * dt;
        float3 r_new = ri + v_new * dt;
        pos_out[i] = float4(r_new, 0.0f);
        vel_out[i] = float4(v_new, 0.0f);
    }
}
```

Incumbent result:
  N1728_M5_steps20: correct, 2.77 ms, 8.4 GFLOPS (useful pairs only) (0.2% of 4500 GFLOPS)
  N4096_M7_steps15: correct, 1.42 ms, 29.0 GFLOPS (useful pairs only) (0.6% of 4500 GFLOPS)
  N10648_M10_steps10: correct, 1.63 ms, 43.8 GFLOPS (useful pairs only) (1.0% of 4500 GFLOPS)
  score (gmean of fraction): 0.0049

## History

- iter  0: compile=OK | correct=True | score=0.0030730385492032603
- iter  1: compile=OK | correct=True | score=0.00330260376920569
- iter  2: compile=OK | correct=True | score=0.004894774396644935
- iter  3: compile=OK | correct=True | score=0.0020042000297635356

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
