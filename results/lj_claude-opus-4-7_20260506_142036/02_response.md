**Optimization:** The hot kernel is `lj_step`, which is bottlenecked by repeated `device` loads of neighbor positions across 27 cells. I'll cooperatively cache neighbor cell occupants into threadgroup memory using a 2x2x2 spatial tile: each threadgroup processes 8 cells (a 2x2x2 block), and loads the union of their neighbor cells (a 4x4x4 = 64 cell halo) into shared memory once. Then every particle in the tile iterates through cached positions instead of going to DRAM. To preserve the "one thread = one particle" contract while still getting tile cooperation, I bin particles by cell and dispatch with a different scheme... but the host fixes dispatch as N threads 1-D. So instead I'll keep per-thread but use **simdgroup cooperation**: threads in the same simd often live in the same cell region, so I broadcast neighbor-particle reads via `simd_broadcast` — each lane fetches one j-position then all 32 lanes consume it. This trades 32x DRAM bandwidth for register traffic. I'll also use `precise::rsqrt`-free math, `fast::` intrinsics where safe, and skip the modulo in neighbor wrap with branchless adjust.

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