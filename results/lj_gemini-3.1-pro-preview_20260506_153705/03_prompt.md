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
    float3 r = pos[i].xyz;
    float inv_L = 1.0f / L;
    r -= L * floor(r * inv_L);
    
    float inv_cell_size = float(M) * inv_L;
    uint cx = min(uint(r.x * inv_cell_size), M - 1u);
    uint cy = min(uint(r.y * inv_cell_size), M - 1u);
    uint cz = min(uint(r.z * inv_cell_size), M - 1u);
    
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
    
    float4 pos_i = pos_in[i];
    float3 ri = pos_i.xyz;
    float3 vi = vel_in[i].xyz;
    
    float inv_L = 1.0f / L;
    float inv_cell_size = float(M) * inv_L;
    
    float3 ri_w = ri - L * floor(ri * inv_L);
    int M_i = int(M);
    int cx = min(int(ri_w.x * inv_cell_size), M_i - 1);
    int cy = min(int(ri_w.y * inv_cell_size), M_i - 1);
    int cz = min(int(ri_w.z * inv_cell_size), M_i - 1);

    int nx[3] = { cx == 0 ? M_i - 1 : cx - 1, cx, cx == M_i - 1 ? 0 : cx + 1 };
    int ny[3] = { cy == 0 ? M_i - 1 : cy - 1, cy, cy == M_i - 1 ? 0 : cy + 1 };
    int nz[3] = { cz == 0 ? M_i - 1 : cz - 1, cz, cz == M_i - 1 ? 0 : cz + 1 };

    float cell_size = L / float(M);
    float rx = ri_w.x - float(cx) * cell_size;
    float ry = ri_w.y - float(cy) * cell_size;
    float rz = ri_w.z - float(cz) * cell_size;

    float3 a = float3(0.0f);

    #pragma unroll
    for (int z = 0; z < 3; ++z) {
        int cz_n = nz[z] * M_i;
        int oz = z - 1;
        float dz_min = (oz == 1) ? max(0.0f, cell_size - rz) : ((oz == -1) ? max(0.0f, rz) : 0.0f);
        float dz_min2 = dz_min * dz_min;

        #pragma unroll
        for (int y = 0; y < 3; ++y) {
            int cy_n = (cz_n + ny[y]) * M_i;
            int oy = y - 1;
            float dy_min = (oy == 1) ? max(0.0f, cell_size - ry) : ((oy == -1) ? max(0.0f, ry) : 0.0f);
            float dzy_min2 = dz_min2 + dy_min * dy_min;

            #pragma unroll
            for (int x = 0; x < 3; ++x) {
                int ox = x - 1;
                float dx_min = (ox == 1) ? max(0.0f, cell_size - rx) : ((ox == -1) ? max(0.0f, rx) : 0.0f);
                float min_dist2 = dzy_min2 + dx_min * dx_min;

                // Geometrically impossible for ANY particle in this neighbor cell to be within rcut
                if (min_dist2 >= rcut2) continue;

                uint nc = uint(cy_n + nx[x]);
                uint cnt = min(cell_count[nc], MAX_PER_CELL);
                if (cnt == 0) continue;

                uint cell_base_scalar = nc * MAX_PER_CELL;
                device const uint4* cell_list_u4 = (device const uint4*)(cell_list + cell_base_scalar);
                uint num_chunks = (cnt + 3) / 4;
                
                for (uint k = 0; k < num_chunks; ++k) {
                    uint4 j4 = cell_list_u4[k];
                    uint4 indices = uint4(k*4) + uint4(0, 1, 2, 3);
                    bool4 valid = indices < uint4(cnt);
                    
                    // Replace out-of-bounds padded slots with self-index (r2 = 0) to avoid memory faults
                    j4.x = valid.x ? j4.x : i;
                    j4.y = valid.y ? j4.y : i;
                    j4.z = valid.z ? j4.z : i;
                    j4.w = valid.w ? j4.w : i;

                    float4 pos0 = pos_in[j4.x];
                    float4 pos1 = pos_in[j4.y];
                    float4 pos2 = pos_in[j4.z];
                    float4 pos3 = pos_in[j4.w];

                    float4 dx = float4(pos0.x, pos1.x, pos2.x, pos3.x) - ri.x;
                    float4 dy = float4(pos0.y, pos1.y, pos2.y, pos3.y) - ri.y;
                    float4 dz = float4(pos0.z, pos1.z, pos2.z, pos3.z) - ri.z;

                    dx -= L * round(dx * inv_L);
                    dy -= L * round(dy * inv_L);
                    dz -= L * round(dz * inv_L);

                    float4 r2 = dx*dx + dy*dy + dz*dz;
                    
                    // Branchless conditional bounds and mathematical safety
                    bool4 mask = (r2 < rcut2) && (r2 > 1e-12f);
                    float4 r2_safe = select(float4(1.0f), r2, mask);
                    
                    float4 inv_r2 = 1.0f / r2_safe;
                    float4 inv_r6 = inv_r2 * inv_r2 * inv_r2;
                    float4 f_mag = -24.0f * inv_r6 * (2.0f * inv_r6 - 1.0f) * inv_r2;
                    
                    f_mag = select(float4(0.0f), f_mag, mask);

                    // Horizontal reduction directly applying forces
                    a.x += dot(f_mag, dx);
                    a.y += dot(f_mag, dy);
                    a.z += dot(f_mag, dz);
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
  N1728_M5_steps20: correct, 2.52 ms, 9.2 GFLOPS (useful pairs only) (0.2% of 4500 GFLOPS)
  N4096_M7_steps15: correct, 2.12 ms, 19.5 GFLOPS (useful pairs only) (0.4% of 4500 GFLOPS)
  N10648_M10_steps10: correct, 1.78 ms, 40.1 GFLOPS (useful pairs only) (0.9% of 4500 GFLOPS)
  score (gmean of fraction): 0.0043

## History

- iter  0: compile=OK | correct=True | score=0.0028461040676860555
- iter  1: compile=OK | correct=True | score=0.004286953578398174
- iter  2: compile=OK | correct=True | score=0.004296831464338089

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
