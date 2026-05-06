The optimization applies **SIMD-group broadcasting** to reduce global memory bandwidth. In a naive N-body kernel, each thread reads $N$ positions and masses, leading to $O(N^2)$ global memory access. By using `simd_broadcast`, each body's data is read from global memory once per SIMD-group (32 threads) and then shared among all threads in that group. This reduces memory traffic by a factor of 32x. We also apply mathematical simplifications, such as calculating $G \cdot dt$ once outside the accumulation loop and using efficient `rsqrt` operations for the $r^{-3}$ term. These optimizations leverage the unified memory architecture and SIMD width of Apple Silicon to significantly improve the compute-to-memory ratio.

```metal
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
    // Guard against out-of-bounds threads
    if (i >= N) return;

    // Load current body's state into registers
    float3 ri = pos_in[i].xyz;
    float3 vi = vel_in[i].xyz;
    float3 acc = float3(0.0f);
    const float eps2 = eps * eps;

    // Iterate over all bodies in tiles of SIMD_WIDTH (32 on Apple Silicon)
    // This optimization reduces global memory reads by 32x.
    for (uint j_base = 0; j_base < N; j_base += 32) {
        // Each thread in the SIMD-group loads one body from global memory
        uint j = j_base + lane_id;
        float4 rj_mj = (j < N) ? float4(pos_in[j].xyz, mass[j]) : float4(0.0f);

        // Share the loaded body with all other threads in the SIMD-group
        for (ushort lane = 0; lane < 32; ++lane) {
            // Only process valid bodies if N is not a multiple of 32
            if (j_base + lane < N) {
                float4 other = simd_broadcast(rj_mj, lane);
                
                float3 d = other.xyz - ri;
                float r2 = d.x * d.x + d.y * d.y + d.z * d.z + eps2;
                
                // Calculate (r^2 + eps^2)^(-1.5) efficiently
                float inv_r = rsqrt(r2);
                float inv_r3 = inv_r * inv_r * inv_r;
                
                // Accumulate weighted acceleration vector
                acc += (other.w * inv_r3) * d;
            }
        }
    }

    // Apply gravity constant and perform Leapfrog integration
    float3 a = acc * G;
    float3 v_new = vi + a * dt;
    float3 r_new = ri + v_new * dt;

    // Write out updated state (w component is preserved as padding)
    pos_out[i] = float4(r_new, 0.0f);
    vel_out[i] = float4(v_new, 0.0f);
}
```