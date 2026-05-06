**Optimization:** Cooperative threadgroup loading of neighbor cells. Each threadgroup processes 32 particles from the same cell (host dispatches by thread index, but particles at consecutive `i` tend to be in similar regions). Actually, the better approach here: use **threadgroup memory to share neighbor positions** — but since particles in a threadgroup may not share cells, that's hard.

Instead, the key insight: the inner loop is **memory-bound** on `pos_in[j]` gathers. I'll (1) **pack cell occupant positions directly into cell_list** would change layout — not allowed. (2) Switch to **scalar arithmetic with float-only ops**, eliminate the `eps` branch (use `s > 0.0f` since `r2==0` only when j==i, and treat that with single mask), (3) use `precise::divide` → `fast` reciprocal via `1.0f/x` with `-ffast-math` semantics already, (4) **hoist the self-skip**: skip when `j==i` rather than checking `r2>0`, removing one comparison per pair, (5) load `pos_in[j]` as `float4` directly (one vectorized load) and combine masks into single select. Reducing per-pair work from ~25 ops to ~18 ops should help the compute-bound inner loop.

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

[[max_total_threads_per_threadgroup(128)]]
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

    float3 ri_w = ri - L * floor(ri * invL);
    int cx = min(int(ri_w.x * inv_cell_size), M_i - 1);
    int cy = min(int(ri_w.y * inv_cell_size), M_i - 1);
    int cz = min(int(ri_w.z * inv_cell_size), M_i - 1);

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
    uint self_i = i;

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

                uint k = 0;

                // Unroll by 8.
                for (; k + 8 <= cnt; k += 8) {
                    uint j0 = list_ptr[k+0];
                    uint j1 = list_ptr[k+1];
                    uint j2 = list_ptr[k+2];
                    uint j3 = list_ptr[k+3];
                    uint j4 = list_ptr[k+4];
                    uint j5 = list_ptr[k+5];
                    uint j6 = list_ptr[k+6];
                    uint j7 = list_ptr[k+7];

                    float4 p0 = pos_in[j0];
                    float4 p1 = pos_in[j1];
                    float4 p2 = pos_in[j2];
                    float4 p3 = pos_in[j3];
                    float4 p4 = pos_in[j4];
                    float4 p5 = pos_in[j5];
                    float4 p6 = pos_in[j6];
                    float4 p7 = pos_in[j7];

                    float3 d0 = p0.xyz - ri; d0 -= L * rint(d0 * invL);
                    float3 d1 = p1.xyz - ri; d1 -= L * rint(d1 * invL);
                    float3 d2 = p2.xyz - ri; d2 -= L * rint(d2 * invL);
                    float3 d3 = p3.xyz - ri; d3 -= L * rint(d3 * invL);
                    float3 d4 = p4.xyz - ri; d4 -= L * rint(d4 * invL);
                    float3 d5 = p5.xyz - ri; d5 -= L * rint(d5 * invL);
                    float3 d6 = p6.xyz - ri; d6 -= L * rint(d6 * invL);
                    float3 d7 = p7.xyz - ri; d7 -= L * rint(d7 * invL);

                    float s0 = dot(d0,d0);
                    float s1 = dot(d1,d1);
                    float s2 = dot(d2,d2);
                    float s3 = dot(d3,d3);
                    float s4 = dot(d4,d4);
                    float s5 = dot(d5,d5);
                    float s6 = dot(d6,d6);
                    float s7 = dot(d7,d7);

                    // Skip self via index, skip non-interacting via cutoff.
                    bool m0 = (j0 != self_i) && (s0 < rcut2);
                    bool m1 = (j1 != self_i) && (s1 < rcut2);
                    bool m2 = (j2 != self_i) && (s2 < rcut2);
                    bool m3 = (j3 != self_i) && (s3 < rcut2);
                    bool m4 = (j4 != self_i) && (s4 < rcut2);
                    bool m5 = (j5 != self_i) && (s5 < rcut2);
                    bool m6 = (j6 != self_i) && (s6 < rcut2);
                    bool m7 = (j7 != self_i) && (s7 < rcut2);

                    float ss0 = m0 ? s0 : 1.0f;
                    float ss1 = m1 ? s1 : 1.0f;
                    float ss2 = m2 ? s2 : 1.0f;
                    float ss3 = m3 ? s3 : 1.0f;
                    float ss4 = m4 ? s4 : 1.0f;
                    float ss5 = m5 ? s5 : 1.0f;
                    float ss6 = m6 ? s6 : 1.0f;
                    float ss7 = m7 ? s7 : 1.0f;

                    float ir2_0 = 1.0f / ss0;
                    float ir2_1 = 1.0f / ss1;
                    float ir2_2 = 1.0f / ss2;
                    float ir2_3 = 1.0f / ss3;
                    float ir2_4 = 1.0f / ss4;
                    float ir2_5 = 1.0f / ss5;
                    float ir2_6 = 1.0f / ss6;
                    float ir2_7 = 1.0f / ss7;

                    float ir6_0 = ir2_0*ir2_0*ir2_0;
                    float ir6_1 = ir2_1*ir2_1*ir2_1;
                    float ir6_2 = ir2_2*ir2_2*ir2_2;
                    float ir6_3 = ir2_3*ir2_3*ir2_3;
                    float ir6_4 = ir2_4*ir2_4*ir2_4;
                    float ir6_5 = ir2_5*ir2_5*ir2_5;
                    float ir6_6 = ir2_6*ir2_6*ir2_6;
                    float ir6_7 = ir2_7*ir2_7*ir2_7;

                    float fm0 = -24.0f * (2.0f*ir6_0*ir6_0 - ir6_0) * ir2_0;
                    float fm1 = -24.0f * (2.0f*ir6_1*ir6_1 - ir6_1) * ir2_1;
                    float fm2 = -24.0f * (2.0f*ir6_2*ir6_2 - ir6_2) * ir2_2;
                    float fm3 = -24.0f * (2.0f*ir6_3*ir6_3 - ir6_3) * ir2_3;
                    float fm4 = -24.0f * (2.0f*ir6_4*ir6_4 - ir6_4) * ir2_4;
                    float fm5 = -24.0f * (2.0f*ir6_5*ir6_5 - ir6_5) * ir2_5;
                    float fm6 = -24.0f * (2.0f*ir6_6*ir6_6 - ir6_6) * ir2_6;
                    float fm7 = -24.0f * (2.0f*ir6_7*ir6_7 - ir6_7) * ir2_7;

                    fm0 = m0 ? fm0 : 0.0f;
                    fm1 = m1 ? fm1 : 0.0f;
                    fm2 = m2 ? fm2 : 0.0f;
                    fm3 = m3 ? fm3 : 0.0f;
                    fm4 = m4 ? fm4 : 0.0f;
                    fm5 = m5 ? fm5 : 0.0f;
                    fm6 = m6 ? fm6 : 0.0f;
                    fm7 = m7 ? fm7 : 0.0f;

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
                    bool m = (j != self_i) && (r2 < rcut2);
                    float ss = m ? r2 : 1.0f;
                    float ir2 = 1.0f / ss;
                    float ir6 = ir2*ir2*ir2;
                    float fm  = -24.0f * (2.0f*ir6*ir6 - ir6) * ir2;
                    fm = m ? fm : 0.0f;
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