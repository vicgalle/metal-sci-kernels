#include <metal_stdlib>
using namespace metal;

kernel void nbody_step(device const float4 *pos_in  [[buffer(0)]],
                       device       float4 *pos_out [[buffer(1)]],
                       device const float4 *vel_in  [[buffer(2)]],
                       device       float4 *vel_out [[buffer(3)]],
                       device const float  *mass    [[buffer(4)]],
                       constant uint        &N      [[buffer(5)]],
                       constant float       &dt     [[buffer(6)]],
                       constant float       &eps    [[buffer(7)]],
                       constant float       &G      [[buffer(8)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= N) return;

    float3 ri = pos_in[i].xyz;
    float3 vi = vel_in[i].xyz;
    float3 a  = float3(0.0f);

    float eps2 = eps * eps;
    
    // Process all bodies in blocks of 32 using threadgroup memory
    for (uint block_start = 0; block_start < N; block_start += 32) {
        uint block_end = min(block_start + 32u, N);
        
        // Load a block of positions and masses into threadgroup memory
        threadgroup float3 pos_block[32];
        threadgroup float  mass_block[32];
        
        for (uint idx = i % 32; idx < (block_end - block_start); idx += 1) {
            uint j = block_start + idx;
            if (j < N) {
                pos_block[idx] = pos_in[j].xyz;
                mass_block[idx] = mass[j];
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute interactions with all bodies in this block
        for (uint idx = 0; idx < (block_end - block_start); ++idx) {
            float3 rj = pos_block[idx];
            float  mj = mass_block[idx];
            float3 d  = rj - ri;
            float  r2 = dot(d, d) + eps2;
            float  inv_r3 = rsqrt(r2 * r2 * r2);
            a += G * mj * d * inv_r3;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float3 v_new = vi + a * dt;
    float3 r_new = ri + v_new * dt;

    pos_out[i] = float4(r_new, 0.0f);
    vel_out[i] = float4(v_new, 0.0f);
}