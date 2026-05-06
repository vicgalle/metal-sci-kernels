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

    int nxs0 = (cx == 0)       ? (M_i - 1) : (cx - 1);
    int nxs1 = cx;
    int nxs2 = (cx == M_i - 1) ? 0         : (cx + 1);
    int nys0 = (cy == 0)       ? (M_i - 1) : (cy - 1);
    int nys1 = cy;
    int nys2 = (cy == M_i - 1) ? 0         : (cy + 1);
    int nzs0 = (cz == 0)       ? (M_i - 1) : (cz - 1);
    int nzs1 = cz;
    int nzs2 = (cz == M_i - 1) ? 0         : (cz + 1);

    int nxs[3] = {nxs0, nxs1, nxs2};
    int nys[3] = {nys0, nys1, nys2};
    int nzs[3] = {nzs0, nzs1, nzs2};

    uint MPC = MAX_PER_CELL;
    float Linv = invL;
    float Lc = L;

    for (int dz = 0; dz < 3; ++dz) {
        int nz = nzs[dz];
        int z_off = nz * M_i;
        for (int dy = 0; dy < 3; ++dy) {
            int ny = nys[dy];
            int row_base = (z_off + ny) * M_i;
            for (int dx = 0; dx < 3; ++dx) {
                int nx_ = nxs[dx];
                uint nc  = uint(row_base + nx_);
                uint cnt = min(cell_count[nc], MPC);
                device const uint *list_ptr = cell_list + nc * MPC;

                uint k = 0;
                // 4-way unroll: keeps register pressure low, high ILP
                for (; k + 4 <= cnt; k += 4) {
                    uint j0 = list_ptr[k+0];
                    uint j1 = list_ptr[k+1];
                    uint j2 = list_ptr[k+2];
                    uint j3 = list_ptr[k+3];

                    float3 r0 = pos_in[j0].xyz;
                    float3 r1 = pos_in[j1].xyz;
                    float3 r2v = pos_in[j2].xyz;
                    float3 r3 = pos_in[j3].xyz;

                    float3 d0 = r0 - ri;  d0 -= Lc * rint(d0 * Linv);
                    float3 d1 = r1 - ri;  d1 -= Lc * rint(d1 * Linv);
                    float3 d2 = r2v - ri; d2 -= Lc * rint(d2 * Linv);
                    float3 d3 = r3 - ri;  d3 -= Lc * rint(d3 * Linv);

                    float s0 = dot(d0,d0);
                    float s1 = dot(d1,d1);
                    float s2 = dot(d2,d2);
                    float s3 = dot(d3,d3);

                    bool m0 = (s0 < rcut2) & (s0 > 0.0f);
                    bool m1 = (s1 < rcut2) & (s1 > 0.0f);
                    bool m2 = (s2 < rcut2) & (s2 > 0.0f);
                    bool m3 = (s3 < rcut2) & (s3 > 0.0f);

                    float ss0 = m0 ? s0 : 1.0f;
                    float ss1 = m1 ? s1 : 1.0f;
                    float ss2 = m2 ? s2 : 1.0f;
                    float ss3 = m3 ? s3 : 1.0f;

                    float ir2_0 = 1.0f / ss0;
                    float ir2_1 = 1.0f / ss1;
                    float ir2_2 = 1.0f / ss2;
                    float ir2_3 = 1.0f / ss3;

                    float ir6_0 = ir2_0*ir2_0*ir2_0;
                    float ir6_1 = ir2_1*ir2_1*ir2_1;
                    float ir6_2 = ir2_2*ir2_2*ir2_2;
                    float ir6_3 = ir2_3*ir2_3*ir2_3;

                    float fm0 = -24.0f * (2.0f*ir6_0*ir6_0 - ir6_0) * ir2_0;
                    float fm1 = -24.0f * (2.0f*ir6_1*ir6_1 - ir6_1) * ir2_1;
                    float fm2 = -24.0f * (2.0f*ir6_2*ir6_2 - ir6_2) * ir2_2;
                    float fm3 = -24.0f * (2.0f*ir6_3*ir6_3 - ir6_3) * ir2_3;

                    fm0 = m0 ? fm0 : 0.0f;
                    fm1 = m1 ? fm1 : 0.0f;
                    fm2 = m2 ? fm2 : 0.0f;
                    fm3 = m3 ? fm3 : 0.0f;

                    a = fma(float3(fm0), d0, a);
                    a = fma(float3(fm1), d1, a);
                    a = fma(float3(fm2), d2, a);
                    a = fma(float3(fm3), d3, a);
                }

                for (; k < cnt; ++k) {
                    uint j = list_ptr[k];
                    float3 rj = pos_in[j].xyz;
                    float3 d  = rj - ri;
                    d -= Lc * rint(d * Linv);
                    float r2 = dot(d, d);
                    bool m = (r2 < rcut2) & (r2 > 0.0f);
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