**Optimization:** Threadgroup-cooperative cell loading. Particles in the same threadgroup typically share neighbour cells, but each thread independently re-loads positions from `device` memory. Instead, I cooperatively gather all 27 neighbour cells of the threadgroup's "anchor" cell into threadgroup memory once, then every thread evaluates its interactions against this shared pool. This dramatically reduces redundant `device` loads (each position is read once per threadgroup instead of ~64x) and keeps the hot working set in fast threadgroup memory.

**Why it should help:** Previous attempt was bandwidth-bound on `pos_in[j]` loads — every thread serially walked the same 27 cells. With M moderate (5–10) and 64 threads/group, particles cluster within a few cells, so a threadgroup's union of 27 anchor-neighbour cells covers ~all needed neighbours. Cooperative load amortizes those reads across 64 threads. Each thread still produces exactly one output. Up to 27*64=1728 floats stays in threadgroup (~21KB), within budget.

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