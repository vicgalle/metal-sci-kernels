To push performance beyond the incumbent, this kernel refines the 4-way vectorization and eliminates redundant safety overheads. 

1. **Eliminate `j_safe` indirection:** Because `cell_list` is either populated with valid particle indices `< N` or remains zero-initialized padding, `j4` strictly contains valid indices. We load positions indiscriminately without bounds-checking indices, which eliminates a 4-way `select` for memory safety. The mathematical result correctly zeros out padded and self-interaction forces later via the `r2 > 0.0f` and `valid_mask` conditions.
2. **Simplified Self-Interaction Mask:** Due to the exact minimum-image convention, `r2` perfectly falls to `0.0f` for identical particles. Relaxing `r2 > 1e-6f` to `r2 > 0.0f` naturally handles self-interactions and garbage padding mapped to particle `0` in a single check.
3. **Factored Force Accumulation:** Reorganizing the Lennard-Jones polynomial pulls the constant `24.0f` entirely out of the innermost loop. We compute `f_mag_base = ir8 * (1.0f - 2.0f * ir6)`, shrinking ALU footprint per pair, and perform a single `a *= 24.0f` broadcast at the end.
4. **Dynamic Loop Unrolling:** Adding `#pragma unroll 4` inside the chunk loop avoids branch/increment overhead for the typically small cell occupancies.

```metal
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
    uint N_u = N;
    if (i >= N_u) return;
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
    uint N_u = N;
    if (i >= N_u) return;
    
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

    float dx_sq[3] = { rx * rx, 0.0f, (cell_size - rx) * (cell_size - rx) };
    float dy_sq[3] = { ry * ry, 0.0f, (cell_size - ry) * (cell_size - ry) };
    float dz_sq[3] = { rz * rz, 0.0f, (cell_size - rz) * (cell_size - rz) };

    uint cx_n[3] = { cx == 0 ? M_u - 1 : cx - 1, cx, cx == M_u - 1 ? 0 : cx + 1 };
    uint cy_n[3] = { cy == 0 ? M_u - 1 : cy - 1, cy, cy == M_u - 1 ? 0 : cy + 1 };
    uint cz_n[3] = { cz == 0 ? M_u - 1 : cz - 1, cz, cz == M_u - 1 ? 0 : cz + 1 };

    float3 a = float3(0.0f);
    
    uint chunks_per_cell = max_per_cell / 4;
    device const uint4* cell_list_u4 = (device const uint4*)cell_list;

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
                if (zyx2 >= rcut2_f) continue;
                
                uint nc = cy_idx + cx_n[x];
                uint cnt = min(cell_count[nc], max_per_cell);
                if (cnt == 0) continue;
                
                uint cell_chunk_base = nc * chunks_per_cell;
                uint num_chunks = (cnt + 3) / 4;
                
                #pragma unroll 4
                for (uint k = 0; k < num_chunks; ++k) {
                    // Padding contains uninitialized 0s. Safe memory load given valid_mask.
                    uint4 j4 = cell_list_u4[cell_chunk_base + k];
                    
                    float4 p0 = pos_in[j4.x];
                    float4 p1 = pos_in[j4.y];
                    float4 p2 = pos_in[j4.z];
                    float4 p3 = pos_in[j4.w];
                    
                    float4 dx_v = float4(p0.x, p1.x, p2.x, p3.x) - ri.x;
                    float4 dy_v = float4(p0.y, p1.y, p2.y, p3.y) - ri.y;
                    float4 dz_v = float4(p0.z, p1.z, p2.z, p3.z) - ri.z;
                    
                    dx_v -= L_f * rint(dx_v * inv_L);
                    dy_v -= L_f * rint(dy_v * inv_L);
                    dz_v -= L_f * rint(dz_v * inv_L);
                    
                    float4 r2_v = dx_v*dx_v + dy_v*dy_v + dz_v*dz_v;
                    
                    uint base_p = k * 4;
                    bool4 valid_mask = uint4(base_p) + uint4(0, 1, 2, 3) < uint4(cnt);
                    
                    // Filter pairs: distance within rcut, ignoring self-interact/padding exactly mapping to 0 distance
                    bool4 mask = (r2_v < rcut2_f) && (r2_v > 0.0f) && valid_mask;
                    
                    // Branchless conditional guard
                    r2_v = select(float4(1.0f), r2_v, mask);
                    
                    float4 ir2 = 1.0f / r2_v;
                    float4 ir6 = ir2 * ir2 * ir2;
                    float4 ir8 = ir6 * ir2;
                    
                    // Inner polynomial evaluated with 24.0f multiplier factored out
                    float4 f_mag_base = ir8 * (1.0f - 2.0f * ir6);
                    float4 f_mag = select(float4(0.0f), f_mag_base, mask);
                    
                    a.x += dot(f_mag, dx_v);
                    a.y += dot(f_mag, dy_v);
                    a.z += dot(f_mag, dz_v);
                }
            }
        }
    }

    a *= 24.0f;
    float3 v_new = vi + a * dt_f;
    float3 r_new = ri + v_new * dt_f;
    
    pos_out[i] = float4(r_new, pos_i.w);
    vel_out[i] = float4(v_new, vel_i.w);
}
```