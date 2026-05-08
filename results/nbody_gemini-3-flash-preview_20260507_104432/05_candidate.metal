#include <metal_stdlib>
using namespace metal;

[[kernel]]
[[max_total_threads_per_threadgroup(256)]]
void nbody_step(device const float4 *pos_in  [[buffer(0)]],
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
    
    // Boundary check
    if (i >= N) return;

    // Cache current body state into registers
    const float3 ri = pos_in[i].xyz;
    const float3 vi = vel_in[i].xyz;
    float3 acc = float3(0.0f);
    const float eps2 = eps * eps;
    
    // Shared memory for cooperative tiling
    threadgroup float4 tile[256];

    // Iterate over all bodies in the system using tiles
    for (uint j_tile = 0; j_tile < N; j_tile += tpg) {
        // Cooperative load: each thread fetches one body into shared memory
        const uint j_idx = j_tile + ti;
        if (j_idx < N) {
            float4 p = pos_in[j_idx];
            // Pre-multiply mass by G here to save N*N multiplications in the loop
            p.w = mass[j_idx] * G;
            tile[ti] = p;
        } else {
            tile[ti] = float4(0.0f);
        }

        // Synchronize threads to ensure the tile is fully loaded
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate acceleration from all bodies in the current tile
        // Unroll inner loop to hide latency of rsqrt and FMA
        #pragma unroll(8)
        for (uint j = 0; j < tpg; ++j) {
            const float4 pj = tile[j];
            const float3 d = pj.xyz - ri;
            
            // Standard N-body gravity calculation with softening
            const float r2 = dot(d, d) + eps2;
            const float inv_r = rsqrt(r2);
            const float inv_r3 = (inv_r * inv_r) * inv_r;
            
            // acc = sum( G * m_j * (r_j - r_i) / |r_j - r_i|^3 )
            acc = fma(d, pj.w * inv_r3, acc);
        }

        // Synchronize before loading the next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Leapfrog Integration: 
    // v_new = v + a * dt
    // r_new = r + v_new * dt
    const float3 v_next = fma(acc, dt, vi);
    const float3 r_next = fma(v_next, dt, ri);

    // Write back results to global memory
    pos_out[i] = float4(r_next, 0.0f);
    vel_out[i] = float4(v_next, 0.0f);
}