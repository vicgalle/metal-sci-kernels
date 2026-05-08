1) I optimized the `lj_step` kernel by manually resolving the 27 neighbor cell index calculations via modulo-free boundary logic and `#pragma unroll`ing the spatial loops. The inner particle-pair loop is unrolled by 2 using vectorized `uint2` fetches from `cell_list` to halve memory requests and overlap memory latency with computation. I replaced the slow division `/ L` with `* inv_L` for minimum-image wrapping (numerically identical since ties near 0.5 wrap to magnitudes well beyond `rcut`). The `j == i` check was removed as `r2 > 1e-12f` intrinsically drops self-interactions.

2) Precomputing the neighbor offsets removes 81 expensive modulo operations and branching per thread. Vectorized memory access maximizes bandwidth. Extensive loop unrolling provides the compiler with massive instruction-level parallelism, allowing independent position fetches and distance calculations to overlap, drastically reducing latency bottlenecks inherent to indirect memory reads.

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
    r -= L * floor(r / L);
    float cell_size = L / float(M);
    uint cx = min(uint(r.x / cell_size), M - 1u);
    uint cy = min(uint(r.y / cell_size), M - 1u);
    uint cz = min(uint(r.z / cell_size), M - 1u);
    uint cell = (cz * M + cy) * M + cx;
    uint slot = atomic_fetch_add_explicit(&cell_count[cell], 1u, memory_order_relaxed);
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
    float4 pos_i = pos_in[i];
    float3 ri = pos_i.xyz;
    float3 vi = vel_in[i].xyz;
    float3 a  = float3(0.0f);

    float cell_size = L / float(M);
    float3 ri_w = ri - L * floor(ri / L);
    int M_i = int(M);
    int cx = min(int(ri_w.x / cell_size), M_i - 1);
    int cy = min(int(ri_w.y / cell_size), M_i - 1);
    int cz = min(int(ri_w.z / cell_size), M_i - 1);

    // Precompute the 3x3x3 neighbor cell coordinates relative to the particle's own cell, handling PBC wraps
    int nx[3] = { cx == 0 ? M_i - 1 : cx - 1, cx, cx == M_i - 1 ? 0 : cx + 1 };
    int ny[3] = { cy == 0 ? M_i - 1 : cy - 1, cy, cy == M_i - 1 ? 0 : cy + 1 };
    int nz[3] = { cz == 0 ? M_i - 1 : cz - 1, cz, cz == M_i - 1 ? 0 : cz + 1 };

    float inv_L = 1.0f / L;

    #pragma unroll
    for (int z = 0; z < 3; ++z) {
        int cell_z = nz[z] * M_i;
        #pragma unroll
        for (int y = 0; y < 3; ++y) {
            int cell_zy = (cell_z + ny[y]) * M_i;
            #pragma unroll
            for (int x = 0; x < 3; ++x) {
                uint nc = uint(cell_zy + nx[x]);
                uint cnt = min(cell_count[nc], MAX_PER_CELL);
                uint cell_base = nc * MAX_PER_CELL;
                
                uint cnt2 = cnt / 2;
                // MAX_PER_CELL is strictly guaranteed to be even (64), so array bounds and uint2 alignment natively map
                device const uint2* cell_list_u2 = (device const uint2*)(cell_list + cell_base);
                
                for (uint k = 0; k < cnt2; ++k) {
                    uint2 j_vec = cell_list_u2[k];
                    
                    float3 rj0 = pos_in[j_vec.x].xyz;
                    float3 rj1 = pos_in[j_vec.y].xyz;
                    
                    float3 d0 = rj0 - ri;
                    float3 d1 = rj1 - ri;
                    
                    d0 -= L * round(d0 * inv_L);
                    d1 -= L * round(d1 * inv_L);
                    
                    float r2_0 = dot(d0, d0);
                    float r2_1 = dot(d1, d1);
                    
                    if (r2_0 < rcut2 && r2_0 > 1e-12f) {
                        float inv_r2  = 1.0f / r2_0;
                        float inv_r6  = inv_r2 * inv_r2 * inv_r2;
                        float inv_r12 = inv_r6 * inv_r6;
                        a += (-24.0f * (2.0f * inv_r12 - inv_r6) * inv_r2) * d0;
                    }
                    if (r2_1 < rcut2 && r2_1 > 1e-12f) {
                        float inv_r2  = 1.0f / r2_1;
                        float inv_r6  = inv_r2 * inv_r2 * inv_r2;
                        float inv_r12 = inv_r6 * inv_r6;
                        a += (-24.0f * (2.0f * inv_r12 - inv_r6) * inv_r2) * d1;
                    }
                }
                
                if (cnt & 1) {
                    uint j0 = cell_list[cell_base + cnt - 1];
                    float3 rj0 = pos_in[j0].xyz;
                    float3 d0 = rj0 - ri;
                    d0 -= L * round(d0 * inv_L);
                    float r2_0 = dot(d0, d0);
                    if (r2_0 < rcut2 && r2_0 > 1e-12f) {
                        float inv_r2  = 1.0f / r2_0;
                        float inv_r6  = inv_r2 * inv_r2 * inv_r2;
                        float inv_r12 = inv_r6 * inv_r6;
                        a += (-24.0f * (2.0f * inv_r12 - inv_r6) * inv_r2) * d0;
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