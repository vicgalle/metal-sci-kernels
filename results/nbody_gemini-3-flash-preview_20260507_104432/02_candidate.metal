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
    
    // Allocate threadgroup memory for cooperative loading of positions and masses
    // A tile size of 256 is efficient for Apple Silicon SIMD width (32)
    threadgroup float4 shared_tile[256];

    float3 ri = 0.0f;
    if (i < N) {
        ri = pos_in[i].xyz;
    }

    float3 acc = 0.0f;
    const float eps2 = eps * eps;

    // Iterate through all bodies in blocks of tpg
    for (uint j_tile = 0; j_tile < N; j_tile += tpg) {
        // Cooperative load: all threads in the group load one body into shared memory
        const uint j_idx = j_tile + ti;
        if (j_idx < N) {
            float4 p = pos_in[j_idx];
            p.w = mass[j_idx];
            shared_tile[ti] = p;
        } else {
            shared_tile[ti] = 0.0f; // Mass 0 ensures no gravitational contribution
        }

        // Synchronize threads to ensure the entire tile is loaded
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate acceleration from the current tile
        if (i < N) {
            #pragma unroll(8)
            for (uint j = 0; j < tpg; ++j) {
                float4 pj = shared_tile[j];
                float3 d = pj.xyz - ri;
                float r2 = d.x * d.x + d.y * d.y + d.z * d.z + eps2;
                float inv_r = rsqrt(r2);
                float inv_r3 = inv_r * inv_r * inv_r;
                acc += (pj.w * inv_r3) * d;
            }
        }

        // Synchronize before loading the next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Update state using Leapfrog integration: v = v + a*dt, r = r + v*dt
    if (i < N) {
        const float3 vi = vel_in[i].xyz;
        const float3 v_next = vi + (G * dt) * acc;
        const float3 r_next = ri + v_next * dt;

        pos_out[i] = float4(r_next, 0.0f);
        vel_out[i] = float4(v_next, 0.0f);
    }
}