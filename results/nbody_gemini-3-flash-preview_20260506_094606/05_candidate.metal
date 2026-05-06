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
                       uint lane_id [[thread_index_in_simdgroup]]) {
    // Standard guard for 1D grid over N bodies
    if (i >= N) return;

    // Load body i's state into registers
    const float3 ri = pos_in[i].xyz;
    const float3 vi = vel_in[i].xyz;
    const float eps2 = eps * eps;
    float3 acc = float3(0.0f);

    // Iterate through all bodies in tiles of 32 (SIMD width)
    for (uint j_base = 0; j_base < N; j_base += 32) {
        const uint j_load = j_base + lane_id;
        
        // Cooperative load: each thread in SIMD-group loads one body
        // and packs its mass into the padding .w component.
        float4 rj_mj = 0.0f;
        if (j_load < N) {
            rj_mj.xyz = pos_in[j_load].xyz;
            rj_mj.w = mass[j_load];
        }

        // Inner interaction loop: broadcast every particle in the tile
        #pragma unroll(32)
        for (ushort l = 0; l < 32; ++l) {
            // Broadcast position and mass to all lanes in the SIMD group
            const float4 other = simd_broadcast(rj_mj, l);
            
            // Calculate displacement vector and squared distance
            const float3 d = other.xyz - ri;
            const float r2 = d.x*d.x + d.y*d.y + d.z*d.z + eps2;
            
            // Fast reciprocal square root for gravitational force calculation
            const float inv_r = rsqrt(r2);
            const float inv_r3 = inv_r * (inv_r * inv_r);
            
            // Mass is zero for out-of-bounds indices, effectively masking them
            const float force_mag = (j_base + l < N) ? (other.w * inv_r3) : 0.0f;
            acc += d * force_mag;
        }
    }

    // Leapfrog Integration:
    // v_new = v + (a * dt) = v + (acc * G * dt)
    // r_new = r + v_new * dt
    const float dt_G = dt * G;
    const float3 v_new = vi + acc * dt_G;
    const float3 r_new = ri + v_new * dt;

    // Preserve .w as padding for consistency
    pos_out[i] = float4(r_new, 0.0f);
    vel_out[i] = float4(v_new, 0.0f);
}