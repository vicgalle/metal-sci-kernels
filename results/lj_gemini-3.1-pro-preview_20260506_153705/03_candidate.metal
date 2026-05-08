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
    int M_i = int(M);
    int cx = clamp(int(r.x * inv_cell_size), 0, M_i - 1);
    int cy = clamp(int(r.y * inv_cell_size), 0, M_i - 1);
    int cz = clamp(int(r.z * inv_cell_size), 0, M_i - 1);
    
    uint cell = (uint(cz) * M + uint(cy)) * M + uint(cx);
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
    
    float4 pos_i = pos_in[i];
    float3 ri = pos_i.xyz;
    float3 vi = vel_in[i].xyz;
    
    float inv_L = 1.0f / L;
    float inv_cell_size = float(M) * inv_L;
    
    float3 ri_w = ri - L * floor(ri * inv_L);
    int M_i = int(M);
    int cx = clamp(int(ri_w.x * inv_cell_size), 0, M_i - 1);
    int cy = clamp(int(ri_w.y * inv_cell_size), 0, M_i - 1);
    int cz = clamp(int(ri_w.z * inv_cell_size), 0, M_i - 1);

    float cell_size = L / float(M);
    float rx = ri_w.x - float(cx) * cell_size;
    float ry = ri_w.y - float(cy) * cell_size;
    float rz = ri_w.z - float(cz) * cell_size;

    float dz_min2[3];
    float dy_min2[3];
    float dx_min2[3];
    int cz_n_arr[3];
    int cy_n_arr[3];
    int cx_n_arr[3];

    for (int o = -1; o <= 1; ++o) {
        float dz = (o == 1) ? max(0.0f, cell_size - rz) : ((o == -1) ? max(0.0f, rz) : 0.0f);
        float dy = (o == 1) ? max(0.0f, cell_size - ry) : ((o == -1) ? max(0.0f, ry) : 0.0f);
        float dx = (o == 1) ? max(0.0f, cell_size - rx) : ((o == -1) ? max(0.0f, rx) : 0.0f);
        dz_min2[o+1] = dz * dz;
        dy_min2[o+1] = dy * dy;
        dx_min2[o+1] = dx * dx;

        int cz_o = cz + o;
        cz_n_arr[o+1] = (cz_o == -1) ? M_i - 1 : ((cz_o == M_i) ? 0 : cz_o);
        int cy_o = cy + o;
        cy_n_arr[o+1] = (cy_o == -1) ? M_i - 1 : ((cy_o == M_i) ? 0 : cy_o);
        int cx_o = cx + o;
        cx_n_arr[o+1] = (cx_o == -1) ? M_i - 1 : ((cx_o == M_i) ? 0 : cx_o);
    }

    float3 a = float3(0.0f);

    for (int z = 0; z < 3; ++z) {
        float z2 = dz_min2[z];
        int cz_n = cz_n_arr[z];

        for (int y = 0; y < 3; ++y) {
            float zy2 = z2 + dy_min2[y];
            if (zy2 >= rcut2) continue;
            int cy_n = cy_n_arr[y];

            for (int x = 0; x < 3; ++x) {
                float zyx2 = zy2 + dx_min2[x];
                if (zyx2 >= rcut2) continue;
                int cx_n = cx_n_arr[x];

                uint nc = (uint(cz_n) * M + uint(cy_n)) * M + uint(cx_n);
                uint cnt = min(cell_count[nc], MAX_PER_CELL);
                
                uint cell_base = nc * MAX_PER_CELL;
                #pragma unroll(4)
                for (uint p = 0; p < cnt; ++p) {
                    uint j = cell_list[cell_base + p];
                    if (j == i) continue;

                    float3 dx_v = pos_in[j].xyz - ri;
                    dx_v -= L * rint(dx_v * inv_L);
                    
                    float r2 = dot(dx_v, dx_v);
                    if (r2 < rcut2) {
                        float inv_r2 = 1.0f / r2;
                        float inv_r6 = inv_r2 * inv_r2 * inv_r2;
                        float f_mag = -24.0f * inv_r6 * (2.0f * inv_r6 - 1.0f) * inv_r2;
                        a += f_mag * dx_v;
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