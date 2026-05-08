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
    
    // Guard against out-of-bounds threads
    if (i >= N) return;

    // Cache local body position and pre-calculate constants
    const float3 ri = pos_in[i].xyz;
    const float eps2 = eps * eps;
    float3 acc = float3(0.0f);
    
    // Shared memory for cooperative tiling
    threadgroup float4 tile[256];

    // Iterate through all bodies in the system using tiling
    for (uint j_tile = 0; j_tile < N; j_tile += tpg) {
        // Cooperative load: one global memory read per body in the tile
        const uint j_idx = j_tile + ti;
        if (j_idx < N) {
            float4 p = pos_in[j_idx];
            p.w = mass[j_idx] * G; // Store G * mass in the padding
            tile[ti] = p;
        } else {
            tile[ti] = float4(0.0f);
        }

        // Barrier to ensure the tile is fully loaded before computation
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate acceleration from bodies in the current tile
        #pragma unroll(16)
        for (uint j = 0; j < tpg; ++j) {
            const float4 pj = tile[j];
            const float3 d = pj.xyz - ri;
            
            // a_i = sum (m_j * d) / (dist^2 + eps^2)^(3/2)
            const float r2 = fma(d.x, d.x, fma(d.y, d.y, fma(d.z, d.z, eps2)));
            const float inv_r = rsqrt(r2);
            const float inv_r3 = inv_r * inv_r * inv_r;
            
            acc = fma(d, pj.w * inv_r3, acc);
        }

        // Barrier to ensure threads finish computation before loading next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Leapfrog Integration:
    // v_new = v_old + a * dt
    // r_new = r_old + v_new * dt
    const float3 vi = vel_in[i].xyz;
    const float3 v_next = fma(acc, dt, vi);
    const float3 r_next = fma(v_next, dt, ri);

    // Synchronous write-back
    pos_out[i] = float4(r_next, 0.0f);
    vel_out[i] = float4(v_next, 0.0f);
}