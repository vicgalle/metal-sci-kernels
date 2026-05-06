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
                       uint tid [[thread_index_in_threadgroup]],
                       uint tgs [[threadgroup_size_in_threadgroup]]) {
    // Guard against out-of-bounds threads
    if (i >= N) return;

    // Load current particle's state
    const float3 ri = pos_in[i].xyz;
    float3 acc = float3(0.0f);
    const float eps2 = eps * eps;
    
    // Tiling parameters
    const uint TILE_SIZE = 256;
    threadgroup float4 tile[TILE_SIZE];

    for (uint j_base = 0; j_base < N; j_base += TILE_SIZE) {
        // Cooperative load: all threads help fill the tile in threadgroup memory
        for (uint l = tid; l < TILE_SIZE; l += tgs) {
            uint j_idx = j_base + l;
            if (j_idx < N) {
                // Combine position and mass into one float4 for shared storage
                tile[l] = float4(pos_in[j_idx].xyz, mass[j_idx]);
            } else {
                // Pad with zero mass to avoid branching in the inner loop
                tile[l] = float4(0.0f);
            }
        }
        
        // Synchronize threads before computing interactions
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute interactions for this tile (compiler is hinted to unroll)
        #pragma unroll(16)
        for (uint j = 0; j < TILE_SIZE; ++j) {
            float4 rj_mj = tile[j];
            float3 d = rj_mj.xyz - ri;
            float r2 = dot(d, d) + eps2;
            float inv_r = rsqrt(r2);
            float inv_r3 = inv_r * inv_r * inv_r;
            
            // Interaction: a_i += m_j * (r_j - r_i) / (r^2 + eps^2)^(3/2)
            acc += (rj_mj.w * inv_r3) * d;
        }

        // Synchronize before loading the next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Integrate: v_new = v + a*dt; r_new = r + v_new*dt
    const float3 vi = vel_in[i].xyz;
    const float3 a = acc * G;
    const float3 v_new = vi + a * dt;
    const float3 r_new = ri + v_new * dt;

    // Write updated state (w is used for padding)
    pos_out[i] = float4(r_new, 0.0f);
    vel_out[i] = float4(v_new, 0.0f);
}