// Naive seed kernel suite for Lennard-Jones MD with cell-list spatial hash.
//
// Three kernels per timestep, dispatched in this order:
//   1) lj_clear_cells: zero the per-cell occupancy counter (M^3 threads).
//   2) lj_build_cells: each particle thread atomically appends its index
//      to its current cell (N threads).
//   3) lj_step:        each particle thread iterates its 27 neighbour
//      cells, computes pair LJ forces (cutoff rcut, minimum-image PBC),
//      and integrates one symplectic-Euler step (N threads).
//
// Storage:
//   buffer 0 (lj_step): const float4 *pos_in   (N bodies; .xyz, .w padding)
//   buffer 1 (lj_step): device float4 *pos_out
//   buffer 2 (lj_step): const float4 *vel_in
//   buffer 3 (lj_step): device float4 *vel_out
//   buffer 4 (lj_step): const uint   *cell_count   (M^3, plain reads here)
//   buffer 5 (lj_step): const uint   *cell_list    (M^3 * MAX_PER_CELL)
//   buffer 6..11      : N, M, L, rcut2, dt, MAX_PER_CELL  (constant scalars)
//
// Lennard-Jones with sigma = epsilon = mass = 1:
//   U(r) = 4 [ (1/r)^12 - (1/r)^6 ]
//   F_on_i = -24 (2/r^12 - 1/r^6) / r^2 * (r_j - r_i)
// Hard cutoff at rcut = 2.5 (force = 0 beyond), minimum-image PBC in a
// cubic box of side L. Cell layout: cell index = (cz*M + cy)*M + cx with
// cell size L/M >= rcut, so 27 neighbour cells cover all interactions.

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
    // Wrap into [0, L).
    r -= L * floor(r / L);
    float cell_size = L / float(M);
    uint cx = min(uint(r.x / cell_size), M - 1u);
    uint cy = min(uint(r.y / cell_size), M - 1u);
    uint cz = min(uint(r.z / cell_size), M - 1u);
    uint cell = (cz * M + cy) * M + cx;
    uint slot = atomic_fetch_add_explicit(&cell_count[cell], 1u,
                                          memory_order_relaxed);
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
    float3 ri = pos_in[i].xyz;
    float3 vi = vel_in[i].xyz;
    float3 a  = float3(0.0f);

    float cell_size = L / float(M);
    float3 ri_w = ri - L * floor(ri / L);          // wrapped into [0, L)
    int M_i = int(M);
    int cx = min(int(ri_w.x / cell_size), M_i - 1);
    int cy = min(int(ri_w.y / cell_size), M_i - 1);
    int cz = min(int(ri_w.z / cell_size), M_i - 1);

    for (int dz = -1; dz <= 1; ++dz) {
        int nz = (cz + dz + M_i) % M_i;
        for (int dy = -1; dy <= 1; ++dy) {
            int ny = (cy + dy + M_i) % M_i;
            for (int dx = -1; dx <= 1; ++dx) {
                int nx_ = (cx + dx + M_i) % M_i;
                uint nc  = uint((nz * M_i + ny) * M_i + nx_);
                uint cnt = min(cell_count[nc], MAX_PER_CELL);
                for (uint k = 0; k < cnt; ++k) {
                    uint j = cell_list[nc * MAX_PER_CELL + k];
                    if (j == i) continue;
                    float3 rj = pos_in[j].xyz;
                    float3 d  = rj - ri;
                    // Minimum-image periodic wrap.
                    d -= L * round(d / L);
                    float r2 = dot(d, d);
                    if (r2 < rcut2 && r2 > 1e-12f) {
                        float inv_r2  = 1.0f / r2;
                        float inv_r6  = inv_r2 * inv_r2 * inv_r2;
                        float inv_r12 = inv_r6 * inv_r6;
                        // F_on_i = -24 (2/r^12 - 1/r^6) / r^2 * (r_j - r_i)
                        float fmag = -24.0f * (2.0f * inv_r12 - inv_r6) * inv_r2;
                        a += fmag * d;
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
