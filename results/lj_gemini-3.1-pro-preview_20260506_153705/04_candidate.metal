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
    
    float inv_cell_size = float(M) * inv_L;
    uint cx = min(uint(r.x * inv_cell_size), M - 1u);
    uint cy = min(uint(r.y * inv_cell_size), M - 1u);
    uint cz = min(uint(r.z * inv_cell_size), M - 1u);
    
    uint cell = (cz * M + cy) * M + cx;
    uint slot = atomic_fetch_add_explicit(&cell_count[cell], 1u, memory_order_relaxed);
    if (slot < MAX_PER_CELL) {
        cell_list[cell * MAX_PER_CELL + slot] = i;
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
    
    float3 ri = pos_in[i].xyz;
    float3 vi = vel_in[i].xyz;
    
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

    #pragma unroll
    for (int z = 0; z < 3; ++z) {
        float z2 = dz_sq[z];
        uint cz_idx = cz_n[z] * M_u * M_u;
        
        #pragma unroll
        for (int y = 0; y < 3; ++y) {
            float zy2 = z2 + dy_sq[y];
            if (zy2 >= rcut2) continue;
            uint cy_idx = cz_idx + cy_n[y] * M_u;
            
            #pragma unroll
            for (int x = 0; x < 3; ++x) {
                float zyx2 = zy2 + dx_sq[x];
                if (zyx2 >= rcut2) continue;
                
                uint nc = cy_idx + cx_n[x];
                uint cnt = min(cell_count[nc], MAX_PER_CELL);
                if (cnt == 0) continue;
                
                uint cell_base = nc * MAX_PER_CELL;
                device const uint4* cell_list_u4 = (device const uint4*)(cell_list + cell_base);
                uint num_chunks = (cnt + 3) / 4;
                
                for (uint k = 0; k < num_chunks; ++k) {
                    uint4 j4 = cell_list_u4[k];
                    uint base_p = k * 4;
                    
                    uint j_safe[4];
                    j_safe[0] = (base_p + 0 < cnt) ? j4.x : i;
                    j_safe[1] = (base_p + 1 < cnt) ? j4.y : i;
                    j_safe[2] = (base_p + 2 < cnt) ? j4.z : i;
                    j_safe[3] = (base_p + 3 < cnt) ? j4.w : i;
                    
                    float3 p_val[4];
                    #pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        p_val[p] = pos_in[j_safe[p]].xyz;
                    }
                    
                    float3 dx[4];
                    float r2[4];
                    #pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        dx[p] = p_val[p] - ri;
                        dx[p] -= L * rint(dx[p] * inv_L);
                        r2[p] = dot(dx[p], dx[p]);
                    }
                    
                    #pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        if (r2[p] < rcut2 && r2[p] > 1e-6f) {
                            float inv_r2 = 1.0f / r2[p];
                            float inv_r6 = inv_r2 * inv_r2 * inv_r2;
                            a += (-24.0f * inv_r6 * (2.0f * inv_r6 - 1.0f) * inv_r2) * dx[p];
                        }
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