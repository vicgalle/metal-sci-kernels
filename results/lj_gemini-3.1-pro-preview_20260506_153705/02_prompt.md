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
    r -= L * floor(r / L);
    float cell_size = L / float(M);
    uint cx = min(uint(r.x / cell_size), M - 1u);
    uint cy = min(uint(r.y / cell_size), M - 1u);
    uint cz = min(uint(r.z / cell_size), M - 1u);
    uint cell = (cz * M + cy) * M + cx;
    uint slot = atomic_fetch_add_explicit(&cell_count[cell], 1u, memory_order_relaxed);
    if (slot < MAX_PER_CELL) {
        cell_list[cell * MAX_PER_CELL + slot] = i;
    }
}

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
    float4 pos_i = pos_in[i];
    float3 ri = pos_i.xyz;
    float3 vi = vel_in[i].xyz;
    float3 a  = float3(0.0f);

    float cell_size = L / float(M);
    float3 ri_w = ri - L * floor(ri / L);
    int M_i = int(M);
    int cx = min(int(ri_w.x / cell_size), M_i - 1);
    int cy = min(int(ri_w.y / cell_size), M_i - 1);
    int cz = min(int(ri_w.z / cell_size), M_i - 1);

    // Precompute the 3x3x3 neighbor cell coordinates relative to the particle's own cell, handling PBC wraps
    int nx[3] = { cx == 0 ? M_i - 1 : cx - 1, cx, cx == M_i - 1 ? 0 : cx + 1 };
    int ny[3] = { cy == 0 ? M_i - 1 : cy - 1, cy, cy == M_i - 1 ? 0 : cy + 1 };
    int nz[3] = { cz == 0 ? M_i - 1 : cz - 1, cz, cz == M_i - 1 ? 0 : cz + 1 };

    float inv_L = 1.0f / L;

    #pragma unroll
    for (int z = 0; z < 3; ++z) {
        int cell_z = nz[z] * M_i;
        #pragma unroll
        for (int y = 0; y < 3; ++y) {
            int cell_zy = (cell_z + ny[y]) * M_i;
            #pragma unroll
            for (int x = 0; x < 3; ++x) {
                uint nc = uint(cell_zy + nx[x]);
                uint cnt = min(cell_count[nc], MAX_PER_CELL);
                uint cell_base = nc * MAX_PER_CELL;
                
                uint cnt2 = cnt / 2;
                // MAX_PER_CELL is strictly guaranteed to be even (64), so array bounds and uint2 alignment natively map
                device const uint2* cell_list_u2 = (device const uint2*)(cell_list + cell_base);
                
                for (uint k = 0; k < cnt2; ++k) {
                    uint2 j_vec = cell_list_u2[k];
                    
                    float3 rj0 = pos_in[j_vec.x].xyz;
                    float3 rj1 = pos_in[j_vec.y].xyz;
                    
                    float3 d0 = rj0 - ri;
                    float3 d1 = rj1 - ri;
                    
                    d0 -= L * round(d0 * inv_L);
                    d1 -= L * round(d1 * inv_L);
                    
                    float r2_0 = dot(d0, d0);
                    float r2_1 = dot(d1, d1);
                    
                    if (r2_0 < rcut2 && r2_0 > 1e-12f) {
                        float inv_r2  = 1.0f / r2_0;
                        float inv_r6  = inv_r2 * inv_r2 * inv_r2;
                        float inv_r12 = inv_r6 * inv_r6;
                        a += (-24.0f * (2.0f * inv_r12 - inv_r6) * inv_r2) * d0;
                    }
                    if (r2_1 < rcut2 && r2_1 > 1e-12f) {
                        float inv_r2  = 1.0f / r2_1;
                        float inv_r6  = inv_r2 * inv_r2 * inv_r2;
                        float inv_r12 = inv_r6 * inv_r6;
                        a += (-24.0f * (2.0f * inv_r12 - inv_r6) * inv_r2) * d1;
                    }
                }
                
                if (cnt & 1) {
                    uint j0 = cell_list[cell_base + cnt - 1];
                    float3 rj0 = pos_in[j0].xyz;
                    float3 d0 = rj0 - ri;
                    d0 -= L * round(d0 * inv_L);
                    float r2_0 = dot(d0, d0);
                    if (r2_0 < rcut2 && r2_0 > 1e-12f) {
                        float inv_r2  = 1.0f / r2_0;
                        float inv_r6  = inv_r2 * inv_r2 * inv_r2;
                        float inv_r12 = inv_r6 * inv_r6;
                        a += (-24.0f * (2.0f * inv_r12 - inv_r6) * inv_r2) * d0;
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

Result of previous attempt:
  N1728_M5_steps20: correct, 2.67 ms, 8.7 GFLOPS (useful pairs only) (0.2% of 4500 GFLOPS)
  N4096_M7_steps15: correct, 1.84 ms, 22.4 GFLOPS (useful pairs only) (0.5% of 4500 GFLOPS)
  N10648_M10_steps10: correct, 1.94 ms, 36.8 GFLOPS (useful pairs only) (0.8% of 4500 GFLOPS)
  score (gmean of fraction): 0.0043

## History

- iter  0: compile=OK | correct=True | score=0.0028461040676860555
- iter  1: compile=OK | correct=True | score=0.004286953578398174

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
