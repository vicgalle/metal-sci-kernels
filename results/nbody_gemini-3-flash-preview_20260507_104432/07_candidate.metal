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
                       uint i [[thread_position_in_grid]],
                       uint ti [[thread_index_in_threadgroup]],
                       uint tpg [[threads_per_threadgroup]]) {
    
    // Each thread handles exactly one output body
    if (i >= N) return;

    // Cache the current body's position in registers
    const float3 ri = pos_in[i].xyz;
    const float eps2 = eps * eps;
    const float Gdt = G * dt;
    float3 acc = 0.0f;

    // Use threadgroup memory for cooperative tiling across the whole group
    // 256 is the maximum recommended threadgroup size for general compute
    threadgroup float4 shared_tile[256];

    // Main N-squared interaction loop processed in tiles
    for (uint j_tile = 0; j_tile < N; j_tile += tpg) {
        
        // Cooperative load: each thread loads one body's pos + mass into tile
        const uint j_idx = j_tile + ti;
        if (j_idx < N) {
            float4 p = pos_in[j_idx];
            p.w = mass[j_idx]; // Compact mass into float4 padding
            shared_tile[ti] = p;
        } else {
            shared_tile[ti] = 0.0f; // Zero mass ensures no acceleration contribution
        }

        // Wait for all threads to synchronize tile data
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate acceleration from every body in the current tile
        // Unroll the loop to improve arithmetic intensity and hide latency
        #pragma unroll(16)
        for (uint j = 0; j < tpg; ++j) {
            const float4 pj = shared_tile[j];
            const float3 d = pj.xyz - ri;
            
            // Efficient r^2 calculation using FMA
            const float r2 = fma(d.x, d.x, fma(d.y, d.y, fma(d.z, d.z, eps2)));
            const float inv_r = rsqrt(r2);
            const float inv_r3 = inv_r * inv_r * inv_r;
            
            // acc = sum( m_j * (r_j - r_i) / r^3 )
            acc = fma(d, pj.w * inv_r3, acc);
        }

        // Synchronize before the next tile overwrite
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Leapfrog Integration:
    // 1. v_new = v + (G * acc) * dt
    // 2. r_new = r + v_new * dt
    const float3 vi = vel_in[i].xyz;
    const float3 v_next = fma(acc, Gdt, vi);
    const float3 r_next = fma(v_next, dt, ri);

    // Final writeback to global memory
    pos_out[i] = float4(r_next, 0.0f);
    vel_out[i] = float4(v_next, 0.0f);
}