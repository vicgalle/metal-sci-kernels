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
    
    // Boundary check as per host requirement
    if (i >= N) return;

    // Registers to store current body state
    const float3 ri = pos_in[i].xyz;
    const float eps2 = eps * eps;
    const float Gdt = G * dt;
    float3 acc = float3(0.0f);
    
    // Threadgroup tiling for data sharing
    threadgroup float4 tile[256];

    // Main N-squared interaction loop
    for (uint j_tile = 0; j_tile < N; j_tile += tpg) {
        // Cooperative load: one body per thread into shared tile
        const uint j_idx = j_tile + ti;
        if (j_idx < N) {
            float4 p = pos_in[j_idx];
            p.w = mass[j_idx];
            tile[ti] = p;
        } else {
            tile[ti] = float4(0.0f);
        }

        // Wait for all threads to finish loading the tile
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate acceleration from all bodies in the current tile
        #pragma unroll(32)
        for (uint j = 0; j < tpg; ++j) {
            const float4 pj = tile[j];
            const float3 d = pj.xyz - ri;
            
            // a_i = sum( m_j * (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2) )
            const float r2 = dot(d, d) + eps2;
            const float inv_r = rsqrt(r2);
            const float inv_r3 = inv_r * (inv_r * inv_r);
            const float s = pj.w * inv_r3;
            
            acc = fma(d, s, acc);
        }

        // Barrier to prevent overwriting tile before all threads are done
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Leapfrog Integration: 
    // v_new = v + (acc * G) * dt
    // r_new = r + v_new * dt
    const float3 vi = vel_in[i].xyz;
    const float3 v_next = fma(acc, Gdt, vi);
    const float3 r_next = fma(v_next, dt, ri);

    // Final writeback
    pos_out[i] = float4(r_next, 0.0f);
    vel_out[i] = float4(v_next, 0.0f);
}