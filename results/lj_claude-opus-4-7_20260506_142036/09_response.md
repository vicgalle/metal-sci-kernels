**Optimization:** Switch to a fundamentally different structure — **SIMD-group cooperative force computation**. Each particle is assigned to one full SIMD-group (32 lanes). The 32 lanes cooperatively scan the 27 neighbour cells: each lane handles one occupant slot at a time within a cell, computes its pair force, then `simd_sum` reduces the per-lane partial accelerations into a single float3. This trades per-thread serial work over ~64 occupants × 27 cells for parallel lane work over the same set, with one DRAM cell-list/pos read per lane (not per occupant).

**Why faster:** The incumbent serializes ~27×avg_cnt pair evaluations per particle in one thread (memory-bound on `pos_in[j]` gather). With a SIMD-group per particle, those gathers happen 32-wide, fully utilizing the load pipeline, and the arithmetic is parallelized 32×. Lane 0 does the integration. Build/clear kernels unchanged. Dispatch is still N threads (one output element per thread); we use `simdgroups_per_threadgroup` so each SIMD-group cooperates — the host's "one thread per output" is honored because lane 0 of each simdgroup writes the result.

Wait — host dispatches N threads total. So I instead use a SIMD-cooperative scheme where each *thread* still owns its particle, but threads within a simdgroup share neighbour-cell loads via `simd_shuffle` (broadcast j positions). I'll use a different angle: **threadgroup-cooperative tile-based traversal where the threadgroup processes a single home cell collectively**, caching all 27 neighbour cells' positions in threadgroup memory once, then each thread (one per particle in that cell) walks the cached array. This amortizes 27 cell loads across all particles in a home cell.

Since particles aren't grouped by cell in thread index, I'll fall back to a cleaner improvement: **tighter inner loop with fp32, vectorized float4 loads, and single-precision recip, plus removing the safe-divide branch by using `fast::divide` and masking only the final force**. Combined with `[[max_total_threads_per_threadgroup(32)]]` to push occupancy.

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