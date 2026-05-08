#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(1024)]]
kernel void wave_step(device const float *u_prev [[buffer(0)]],
                      device const float *u_curr [[buffer(1)]],
                      device       float *u_next [[buffer(2)]],
                      constant uint      &NX     [[buffer(3)]],
                      constant uint      &NY     [[buffer(4)]],
                      constant uint      &NZ     [[buffer(5)]],
                      constant float     &alpha  [[buffer(6)]],
                      uint3 gid     [[thread_position_in_grid]],
                      uint3 t_pos   [[thread_position_in_threadgroup]],
                      uint3 tg_size [[threads_per_threadgroup]]) {
    uint i = gid.x;
    uint j = gid.y;
    uint k = gid.z;
    
    uint nx = NX;
    uint ny = NY;
    uint nz = NZ;
    
    bool valid = (i < nx && j < ny && k < nz);
    uint idx = (k * ny + j) * nx + i;
    
    // Read the current cell, safely padding with 0.0 if thread is mapped outside the physical grid
    float c = valid ? u_curr[idx] : 0.0f;
    
    // Flattened thread index within the threadgroup block
    uint tid = (t_pos.z * tg_size.y + t_pos.y) * tg_size.x + t_pos.x;
    
    threadgroup float smem[1024];
    smem[tid] = c;
    
    // Ensure all threads in the group have populated their values
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Safely exit threads outside the grid only AFTER the barrier to avoid deadlocks
    if (!valid) return;

    // Dirichlet boundary condition: boundary cells copy unchanged
    bool is_boundary = (i == 0 || j == 0 || k == 0 || 
                        i == nx - 1 || j == ny - 1 || k == nz - 1);
    if (is_boundary) {
        u_next[idx] = c;
        return;
    }

    // Fetch missing neighbors from threadgroup memory if they belong to this threadgroup's block.
    // Otherwise, fallback to global memory (guaranteed to be safely in-bounds here).
    float xm = (t_pos.x > 0) ? smem[tid - 1] : u_curr[idx - 1u];
    float xp = (t_pos.x < tg_size.x - 1u) ? smem[tid + 1] : u_curr[idx + 1u];
    float lap = xm + xp;
    
    float ym = (t_pos.y > 0) ? smem[tid - tg_size.x] : u_curr[idx - nx];
    float yp = (t_pos.y < tg_size.y - 1u) ? smem[tid + tg_size.x] : u_curr[idx + nx];
    lap += ym + yp;
    
    uint stride_z = nx * ny;
    uint tg_stride_z = tg_size.x * tg_size.y;
    float zm = (t_pos.z > 0) ? smem[tid - tg_stride_z] : u_curr[idx - stride_z];
    float zp = (t_pos.z < tg_size.z - 1u) ? smem[tid + tg_stride_z] : u_curr[idx + stride_z];
    lap += zm + zp;
    
    lap -= 6.0f * c;
    
    float a = alpha;
    u_next[idx] = 2.0f * c - u_prev[idx] + a * lap;
}