To achieve high throughput on Apple Silicon, we must minimize divergent arithmetic and unnecessary memory accesses. The previous attempt manually vectorized across 4 particles per thread using `float4`. This forced the GPU to evaluate `rint` and distance checks for *all* 4 particles, even if they were outside the interaction cutoff (which >85% of them are). It also suffered high register pressure.

This optimized version:
1. **Skipping Entire Cells (AABB check)**: Calculates the minimal squared distance (`dx_sq`, etc.) from the particle to the faces of the 27 neighbor cells. We early-exit the Z, Y, and X loops if the neighbor cell boundary is beyond `rcut2`. This avoids loading and evaluating particles for ~50% of the cells entirely.
2. **Scalar Branching**: Replaces `float4` masks with a tight scalar loop. Since <15% of processed pairs fall within the cutoff sphere, `if (r2 < rcut2)` elegantly skips the heavy force accumulation (divisions, mul-adds), mapping efficiently to the GPU's instruction pipeline.
3. **Coalesced Loads**: Explicitly loads `float4` to guarantee single 128-bit memory transactions from `pos_in`.

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
    if (i >= N) return;
    
    float3 r = pos[i].xyz;
    float inv_L = 1.0f / L;
    
    // Wrap to [0, L)
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
    
    float L_f = L;
    uint M_u = M;
    float rcut2_f = rcut2;
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

    // Minimum squared distances to the 3 adjacent cell boundaries per axis
    float dx_sq[3] = { rx * rx, 0.0f, (cell_size - rx) * (cell_size - rx) };
    float dy_sq[3] = { ry * ry, 0.0f, (cell_size - ry) * (cell_size - ry) };
    float dz_sq[3] = { rz * rz, 0.0f, (cell_size - rz) * (cell_size - rz) };

    uint cx_n[3];
    uint cy_n[3];
    uint cz_n[3];

    #pragma unroll
    for (int d = 0; d < 3; ++d) {
        int nx = int(cx) + d - 1;
        cx_n[d] = (nx < 0) ? (M_u - 1u) : ((nx >= int(M_u)) ? 0u : uint(nx));

        int ny = int(cy) + d - 1;
        cy_n[d] = (ny < 0) ? (M_u - 1u) : ((ny >= int(M_u)) ? 0u : uint(ny));

        int nz = int(cz) + d - 1;
        cz_n[d] = (nz < 0) ? (M_u - 1u) : ((nz >= int(M_u)) ? 0u : uint(nz));
    }

    float3 a = float3(0.0f);

    #pragma unroll
    for (int z = 0; z < 3; ++z) {
        float dz2 = dz_sq[z];
        uint cz_i = cz_n[z];
        
        #pragma unroll
        for (int y = 0; y < 3; ++y) {
            float dy2 = dy_sq[y];
            if (dz2 + dy2 >= rcut2_f) continue;
            uint cy_i = cy_n[y];
            
            #pragma unroll
            for (int x = 0; x < 3; ++x) {
                float dx2 = dx_sq[x];
                // Minkowski AABB vs cutoff sphere early rejection
                if (dz2 + dy2 + dx2 >= rcut2_f) continue;
                uint cx_i = cx_n[x];
                
                uint cell_idx = (cz_i * M_u + cy_i) * M_u + cx_i;
                uint count = min(cell_count[cell_idx], max_per_cell);
                uint cell_base = cell_idx * max_per_cell;
                
                #pragma unroll 4
                for (uint k = 0; k < count; ++k) {
                    uint j = cell_list[cell_base + k];
                    
                    // Force 128-bit coalesced memory load
                    float4 pj = pos_in[j];
                    float3 d_vec = pj.xyz - ri;
                    
                    // Periodic minimum-image
                    d_vec -= L_f * rint(d_vec * inv_L);
                    
                    float r2 = dot(d_vec, d_vec);
                    
                    // Predictable scalar branch drastically reduces work footprint
                    if (r2 < rcut2_f && r2 > 1e-6f) {
                        float ir2 = 1.0f / r2;
                        float ir6 = ir2 * ir2 * ir2;
                        float f_mag = ir6 * ir2 * (24.0f - 48.0f * ir6);
                        a += d_vec * f_mag;
                    }
                }
            }
        }
    }

    float dt_f = dt;
    float3 v_new = vi + a * dt_f;
    float3 r_new = ri + v_new * dt_f;
    
    pos_out[i] = float4(r_new, pos_i.w);
    vel_out[i] = float4(v_new, vel_i.w);
}
```