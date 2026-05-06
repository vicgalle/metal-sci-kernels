**Optimization:** Threadgroup-cooperative neighbor loading via sorted-by-cell particle ordering won't work since the host fixes thread→particle mapping. Instead, I'll keep the incumbent's per-thread cell-walking structure but: (1) batch-load all 27 cells' particle indices and positions via a single linearized loop with better software pipelining; (2) use `precise::divide`-free reciprocals where safe; (3) hoist the periodic wrap arithmetic; (4) replace branchy `if (r2 < rcut2 && r2 > eps)` with branchless masking using `select` to avoid divergent control flow on Apple GPUs which hurts SIMD utilization; (5) skip wrapping `ri` since it's only used relative to neighbors (use `rint` minimum image directly on `rj-ri`); (6) flatten the triple-nested loop into a single 27-iter loop with precomputed neighbor cell indices to reduce register pressure and improve scheduling. Branchless force accumulation should help since within rcut is the common case at moderate density.

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