#include <metal_stdlib>
using namespace metal;

inline float3 compute_force(float4 p, float3 ri, float inv_L, float L_f, float rcut2_f) {
    float3 d = p.xyz - ri;
    // Apply minimum image convention
    d -= L_f * rint(d * inv_L);
    float r2 = dot(d, d);
    // Fast path: fully branch out of ALU for rejected or self-interacting pairs
    if (r2 < rcut2_f && r2 > 1e-6f) {
        float ir2 = 1.0f / r2;
        float ir6 = ir2 * ir2 * ir2;
        // Formula equivalent to: -24 * (2 * ir12 - ir6) * ir2 * d
        return (ir6 * ir2 * (24.0f - 48.0f * ir6)) * d;
    }
    return float3(0.0f);
}

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
    float L_f = L;
    uint M_u = M;
    float inv_L = 1.0f / L_f;
    r -= L_f * floor(r * inv_L);
    
    float inv_cell_size = float(M_u) * inv_L;
    uint cx = min(uint(r.x * inv_cell_size), M_u - 1u);
    uint cy = min(uint(r.y * inv_cell_size), M_u - 1u);
    uint cz = min(uint(r.z * inv_cell_size), M_u - 1u);
    
    uint cell = (cz * M_u + cy) * M_u + cx;
    uint max_per_cell = MAX_PER_CELL;
    uint slot = atomic_fetch_add_explicit(&cell_count[cell], 1u, memory_order_relaxed);
    
    if (slot < max_per_cell) {
        cell_list[cell * max_per_cell + slot] = i;
    }
}

[[max_total_threads_per_threadgroup(256)]]
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
    
    uint M_u = M;
    float L_f = L;
    float rcut2_f = rcut2;
    float dt_f = dt;
    uint max_per_cell = MAX_PER_CELL;
    
    float4 pos_i = pos_in[i];
    float4 vel_i = vel_in[i];
    float3 ri = pos_i.xyz;
    float3 vi = vel_i.xyz;
    
    float inv_L = 1.0f / L_f;
    float inv_cell_size = float(M_u) * inv_L;
    
    float3 ri_w = ri - L_f * floor(ri * inv_L);
    uint cx = min(uint(ri_w.x * inv_cell_size), M_u - 1u);
    uint cy = min(uint(ri_w.y * inv_cell_size), M_u - 1u);
    uint cz = min(uint(ri_w.z * inv_cell_size), M_u - 1u);

    float cell_size = L_f / float(M_u);
    float rx = ri_w.x - float(cx) * cell_size;
    float ry = ri_w.y - float(cy) * cell_size;
    float rz = ri_w.z - float(cz) * cell_size;

    // Minimum squared bounding-box distance to the 3x3x3 neighbor cells
    float dx_sq[3] = { rx * rx, 0.0f, (cell_size - rx) * (cell_size - rx) };
    float dy_sq[3] = { ry * ry, 0.0f, (cell_size - ry) * (cell_size - ry) };
    float dz_sq[3] = { rz * rz, 0.0f, (cell_size - rz) * (cell_size - rz) };

    uint cx_n[3] = { cx == 0 ? M_u - 1 : cx - 1, cx, cx == M_u - 1 ? 0 : cx + 1 };
    uint cy_n[3] = { cy == 0 ? M_u - 1 : cy - 1, cy, cy == M_u - 1 ? 0 : cy + 1 };
    uint cz_n[3] = { cz == 0 ? M_u - 1 : cz - 1, cz, cz == M_u - 1 ? 0 : cz + 1 };

    float3 a = float3(0.0f);
    
    uint chunks_per_cell = max_per_cell / 4;
    device const uint4* cell_list_u4 = (device const uint4*)cell_list;

    // Fully unrolled nested loops guarantee registers are used for coordinate math (avoids local memory spills)
    #pragma unroll
    for (int z = 0; z < 3; ++z) {
        float z2 = dz_sq[z];
        uint cz_idx = cz_n[z] * M_u * M_u;
        
        #pragma unroll
        for (int y = 0; y < 3; ++y) {
            float zy2 = z2 + dy_sq[y];
            if (zy2 >= rcut2_f) continue;
            uint cy_idx = cz_idx + cy_n[y] * M_u;
            
            #pragma unroll
            for (int x = 0; x < 3; ++x) {
                float zyx2 = zy2 + dx_sq[x];
                // Spatial cell rejection limits unnecessary loops perfectly
                if (zyx2 >= rcut2_f) continue;
                
                uint nc = cy_idx + cx_n[x];
                uint cnt = min(cell_count[nc], max_per_cell);
                if (cnt == 0) continue;
                
                uint chunks = cnt / 4;
                uint cell_chunk_base = nc * chunks_per_cell;
                
                // Pack sequential fetches up to instruction parallelism limit to hide VRAM latency
                for (uint c = 0; c < chunks; ++c) {
                    uint4 j4 = cell_list_u4[cell_chunk_base + c];
                    
                    float4 p0 = pos_in[j4.x];
                    float4 p1 = pos_in[j4.y];
                    float4 p2 = pos_in[j4.z];
                    float4 p3 = pos_in[j4.w];
                    
                    a += compute_force(p0, ri, inv_L, L_f, rcut2_f);
                    a += compute_force(p1, ri, inv_L, L_f, rcut2_f);
                    a += compute_force(p2, ri, inv_L, L_f, rcut2_f);
                    a += compute_force(p3, ri, inv_L, L_f, rcut2_f);
                }
                
                // Clean tail loop for remaining loads
                uint cell_base = nc * max_per_cell;
                for (uint k = chunks * 4; k < cnt; ++k) {
                    uint j = cell_list[cell_base + k];
                    a += compute_force(pos_in[j], ri, inv_L, L_f, rcut2_f);
                }
            }
        }
    }

    // Symplectic Euler integration update
    float3 v_new = vi + a * dt_f;
    float3 r_new = ri + v_new * dt_f;
    
    pos_out[i] = float4(r_new, pos_i.w);
    vel_out[i] = float4(v_new, vel_i.w);
}