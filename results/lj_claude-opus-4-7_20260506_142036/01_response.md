**Optimization:** The hot kernel is `lj_step`, which is memory-bound on neighbor reads. The biggest win is reducing global-memory traffic by **cooperatively staging neighbor cell data into threadgroup memory** when threads in the same threadgroup share neighborhoods. However, since particles within the same threadgroup may be in different cells, a simpler but very effective approach is:

1. **Pack cell data**: store `float4(pos.xyz, particle_index_as_float_bits)` into a precomputed buffer — but we don't have that buffer available.
2. **Best practical wins**: (a) Use `fast::` math (rsqrt-style), (b) hoist invariants, (c) use `precise::` only where needed, (d) replace expensive `round()` with `rint`/`floor(x+0.5)`, (e) avoid the `j==i` branch by letting r2≈0 skip handle it, (f) use `[[max_total_threads_per_threadgroup]]`, (g) read positions via `float4` once, and (h) unroll the 27-cell loop with `#pragma unroll` hints.
3. For `lj_clear_cells` and `lj_build_cells`, keep them simple — they're tiny.

The dominant cost is the inner loop's memory loads and FMA chain; tightening it with `fma`, removing the self-check branch (the r2≈0 guard handles it), and using `metal::fast` reciprocals should give a solid speedup.

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
    float Mf   = float(M);
    float cell_size = L / Mf;
    float inv_cell_size = Mf * invL;

    // Wrap ri into [0, L) for cell determination only.
    float3 ri_w = ri - L * floor(ri * invL);
    int M_i = int(M);
    int cx = min(int(ri_w.x * inv_cell_size), M_i - 1);
    int cy = min(int(ri_w.y * inv_cell_size), M_i - 1);
    int cz = min(int(ri_w.z * inv_cell_size), M_i - 1);

    // Precompute wrapped neighbor coordinate arrays.
    int nxs[3], nys[3], nzs[3];
    #pragma unroll
    for (int d = -1; d <= 1; ++d) {
        int idx = d + 1;
        nxs[idx] = (cx + d + M_i) % M_i;
        nys[idx] = (cy + d + M_i) % M_i;
        nzs[idx] = (cz + d + M_i) % M_i;
    }

    uint MPC = MAX_PER_CELL;

    #pragma unroll
    for (int dz = 0; dz < 3; ++dz) {
        int nz = nzs[dz];
        #pragma unroll
        for (int dy = 0; dy < 3; ++dy) {
            int ny = nys[dy];
            #pragma unroll
            for (int dx = 0; dx < 3; ++dx) {
                int nx_ = nxs[dx];
                uint nc  = uint((nz * M_i + ny) * M_i + nx_);
                uint cnt = min(cell_count[nc], MPC);
                device const uint *list_ptr = cell_list + nc * MPC;
                for (uint k = 0; k < cnt; ++k) {
                    uint j = list_ptr[k];
                    float3 rj = pos_in[j].xyz;
                    float3 d  = rj - ri;
                    // Minimum-image periodic wrap.
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

    float3 v_new = vi + a * dt;
    float3 r_new = ri + v_new * dt;
    pos_out[i] = float4(r_new, 0.0f);
    vel_out[i] = float4(v_new, 0.0f);
}
```