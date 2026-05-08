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
    
    uint M_u = M;
    float inv_cell_size = float(M_u) * inv_L;
    uint cx = min(uint(r.x * inv_cell_size), M_u - 1u);
    uint cy = min(uint(r.y * inv_cell_size), M_u - 1u);
    uint cz = min(uint(r.z * inv_cell_size), M_u - 1u);
    
    uint cell = (cz * M_u + cy) * M_u + cx;
    uint max_per_cell = MAX_PER_CELL;
    
    uint slot = atomic_fetch_add_explicit(&cell_count[cell], 1u, memory_order_relaxed);
    if (slot < max_per_cell) {
        cell_list[cell * max_per_cell + slot] = i;
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
    float4 vel_i = vel_in[i];
    float3 ri = pos_i.xyz;
    
    float inv_L = 1.0f / L;
    uint M_u = M;
    float inv_cell_size = float(M_u) * inv_L;
    
    float3 ri_w = ri - L * floor(ri * inv_L);
    uint cx = min(uint(ri_w.x * inv_cell_size), M_u - 1u);
    uint cy = min(uint(ri_w.y * inv_cell_size), M_u - 1u);
    uint cz = min(uint(ri_w.z * inv_cell_size), M_u - 1u);

    float cell_size = L / float(M_u);
    float rx = ri_w.x - float(cx) * cell_size;
    float ry = ri_w.y - float(cy) * cell_size;
    float rz = ri_w.z - float(cz) * cell_size;

    float dx_sq[3] = { rx * rx, 0.0f, (cell_size - rx) * (cell_size - rx) };
    float dy_sq[3] = { ry * ry, 0.0f, (cell_size - ry) * (cell_size - ry) };
    float dz_sq[3] = { rz * rz, 0.0f, (cell_size - rz) * (cell_size - rz) };

    uint cx_n[3] = { cx == 0 ? M_u - 1 : cx - 1, cx, cx == M_u - 1 ? 0 : cx + 1 };
    uint cy_n[3] = { cy == 0 ? M_u - 1 : cy - 1, cy, cy == M_u - 1 ? 0 : cy + 1 };
    uint cz_n[3] = { cz == 0 ? M_u - 1 : cz - 1, cz, cz == M_u - 1 ? 0 : cz + 1 };

    float3 a = float3(0.0f);
    float rcut2_f = rcut2;
    uint max_per_cell = MAX_PER_CELL;
    uint M2 = M_u * M_u;

    #pragma unroll
    for (uint z = 0; z < 3; ++z) {
        float z2 = dz_sq[z];
        uint cz_idx = cz_n[z] * M2;
        
        #pragma unroll
        for (uint y = 0; y < 3; ++y) {
            float zy2 = z2 + dy_sq[y];
            if (zy2 >= rcut2_f) continue;
            uint cy_idx = cz_idx + cy_n[y] * M_u;
            
            #pragma unroll
            for (uint x = 0; x < 3; ++x) {
                float zyx2 = zy2 + dx_sq[x];
                if (zyx2 >= rcut2_f) continue;
                
                uint nc = cy_idx + cx_n[x];
                uint cnt = min(cell_count[nc], max_per_cell);
                
                uint cell_base = nc * max_per_cell;
                for (uint k = 0; k < cnt; ++k) {
                    uint j = cell_list[cell_base + k];
                    
                    float3 pj = pos_in[j].xyz;
                    float3 d = pj - ri;
                    d -= L * rint(d * inv_L);
                    float r2 = dot(d, d);
                    
                    if (r2 < rcut2_f && r2 > 1e-6f) {
                        float ir2 = 1.0f / r2;
                        float ir6 = ir2 * ir2 * ir2;
                        float f_mag = ir6 * ir2 * (24.0f - 48.0f * ir6);
                        a += f_mag * d;
                    }
                }
            }
        }
    }

    float3 v_new = vel_i.xyz + a * dt;
    float3 r_new = ri + v_new * dt;
    pos_out[i] = float4(r_new, pos_i.w);
    vel_out[i] = float4(v_new, vel_i.w);
}
```

Result of previous attempt:
  N1728_M5_steps20: correct, 3.29 ms, 7.1 GFLOPS (useful pairs only) (0.2% of 4500 GFLOPS)
  N4096_M7_steps15: correct, 2.18 ms, 18.9 GFLOPS (useful pairs only) (0.4% of 4500 GFLOPS)
  N10648_M10_steps10: correct, 2.50 ms, 28.6 GFLOPS (useful pairs only) (0.6% of 4500 GFLOPS)
  score (gmean of fraction): 0.0035

## Current best (incumbent)

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
    uint N_u = N;
    if (i >= N_u) return;
    float3 r = pos[i].xyz;
    float L_f = L;
    uint M_u = M;
    float inv_L = 1.0f / L_f;
    r -= L_f * floor(r * inv_L);
    
    float inv_cell_size = float(M_u) * inv_L;
    uint cx = min(uint(r.x * inv_cell_size), M_u - 1u);
    uint cy = min(uint(r.y * inv_cell_size), M_u - 1u);
    uint cz = min(uint(r.z * inv_cell_size), M_u - 1u);
    
    uint cell = (cz * M_u + cy) * M_u + cx;
    uint max_per_cell = MAX_PER_CELL;
    uint slot = atomic_fetch_add_explicit(&cell_count[cell], 1u, memory_order_relaxed);
    if (slot < max_per_cell) {
        cell_list[cell * max_per_cell + slot] = i;
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
    uint N_u = N;
    if (i >= N_u) return;
    
    uint M_u = M;
    float L_f = L;
    float rcut2_f = rcut2;
    float dt_f = dt;
    uint max_per_cell = MAX_PER_CELL;
    
    float4 pos_i = pos_in[i];
    float4 vel_i = vel_in[i];
    float3 ri = pos_i.xyz;
    float3 vi = vel_i.xyz;
    
    float inv_L = 1.0f / L_f;
    float inv_cell_size = float(M_u) * inv_L;
    
    float3 ri_w = ri - L_f * floor(ri * inv_L);
    uint cx = min(uint(ri_w.x * inv_cell_size), M_u - 1u);
    uint cy = min(uint(ri_w.y * inv_cell_size), M_u - 1u);
    uint cz = min(uint(ri_w.z * inv_cell_size), M_u - 1u);

    float cell_size = L_f / float(M_u);
    float rx = ri_w.x - float(cx) * cell_size;
    float ry = ri_w.y - float(cy) * cell_size;
    float rz = ri_w.z - float(cz) * cell_size;

    float dx_sq[3] = { rx * rx, 0.0f, (cell_size - rx) * (cell_size - rx) };
    float dy_sq[3] = { ry * ry, 0.0f, (cell_size - ry) * (cell_size - ry) };
    float dz_sq[3] = { rz * rz, 0.0f, (cell_size - rz) * (cell_size - rz) };

    uint cx_n[3] = { cx == 0 ? M_u - 1 : cx - 1, cx, cx == M_u - 1 ? 0 : cx + 1 };
    uint cy_n[3] = { cy == 0 ? M_u - 1 : cy - 1, cy, cy == M_u - 1 ? 0 : cy + 1 };
    uint cz_n[3] = { cz == 0 ? M_u - 1 : cz - 1, cz, cz == M_u - 1 ? 0 : cz + 1 };

    float3 a = float3(0.0f);
    
    uint chunks_per_cell = max_per_cell / 4;
    device const uint4* cell_list_u4 = (device const uint4*)cell_list;

    #pragma unroll
    for (int z = 0; z < 3; ++z) {
        float z2 = dz_sq[z];
        uint cz_idx = cz_n[z] * M_u * M_u;
        
        #pragma unroll
        for (int y = 0; y < 3; ++y) {
            float zy2 = z2 + dy_sq[y];
            if (zy2 >= rcut2_f) continue;
            uint cy_idx = cz_idx + cy_n[y] * M_u;
            
            #pragma unroll
            for (int x = 0; x < 3; ++x) {
                float zyx2 = zy2 + dx_sq[x];
                if (zyx2 >= rcut2_f) continue;
                
                uint nc = cy_idx + cx_n[x];
                uint cnt = min(cell_count[nc], max_per_cell);
                if (cnt == 0) continue;
                
                uint cell_chunk_base = nc * chunks_per_cell;
                uint num_chunks = (cnt + 3) / 4;
                
                for (uint k = 0; k < num_chunks; ++k) {
                    uint4 j4 = cell_list_u4[cell_chunk_base + k];
                    uint base_p = k * 4;
                    
                    bool4 valid_mask = uint4(base_p) + uint4(0, 1, 2, 3) < uint4(cnt);
                    uint4 j_safe = select(uint4(i), j4, valid_mask);
                    
                    float4 p0 = pos_in[j_safe.x];
                    float4 p1 = pos_in[j_safe.y];
                    float4 p2 = pos_in[j_safe.z];
                    float4 p3 = pos_in[j_safe.w];
                    
                    float4 dx_v = float4(p0.x, p1.x, p2.x, p3.x) - ri.x;
                    float4 dy_v = float4(p0.y, p1.y, p2.y, p3.y) - ri.y;
                    float4 dz_v = float4(p0.z, p1.z, p2.z, p3.z) - ri.z;
                    
                    dx_v -= L_f * rint(dx_v * inv_L);
                    dy_v -= L_f * rint(dy_v * inv_L);
                    dz_v -= L_f * rint(dz_v * inv_L);
                    
                    float4 r2_v = dx_v*dx_v + dy_v*dy_v + dz_v*dz_v;
                    
                    // Identify invalid connections/self-interactions
                    bool4 mask = (r2_v < rcut2_f) && (r2_v > 1e-6f);
                    
                    // Shield div-by-0 for invalid lanes
                    r2_v = select(float4(1.0f), r2_v, mask);
                    
                    float4 ir2 = 1.0f / r2_v;
                    float4 ir6 = ir2 * ir2 * ir2;
                    
                    // Evaluate force magnitude coefficients (masking out ignored lanes)
                    float4 f_mag = select(float4(0.0f), ir6 * ir2 * (24.0f - 48.0f * ir6), mask);
                    
                    a.x += dot(f_mag, dx_v);
                    a.y += dot(f_mag, dy_v);
                    a.z += dot(f_mag, dz_v);
                }
            }
        }
    }

    float3 v_new = vi + a * dt_f;
    float3 r_new = ri + v_new * dt_f;
    pos_out[i] = float4(r_new, pos_i.w);
    vel_out[i] = float4(v_new, vel_i.w);
}
```

Incumbent result:
  N1728_M5_steps20: correct, 2.37 ms, 9.8 GFLOPS (useful pairs only) (0.2% of 4500 GFLOPS)
  N4096_M7_steps15: correct, 1.31 ms, 31.6 GFLOPS (useful pairs only) (0.7% of 4500 GFLOPS)
  N10648_M10_steps10: correct, 1.36 ms, 52.5 GFLOPS (useful pairs only) (1.2% of 4500 GFLOPS)
  score (gmean of fraction): 0.0056

## History

- iter  5: compile=OK | correct=True | score=0.0037016542173682575
- iter  6: compile=FAIL | correct=False | score=N/A
- iter  7: compile=FAIL | correct=False | score=N/A
- iter  8: compile=OK | correct=True | score=0.00563099114126572
- iter  9: compile=OK | correct=True | score=0.004833900420089567
- iter 10: compile=OK | correct=True | score=0.0035434526930920368
- iter 11: compile=OK | correct=True | score=0.005117012684028163
- iter 12: compile=OK | correct=True | score=0.00347697609575754

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
