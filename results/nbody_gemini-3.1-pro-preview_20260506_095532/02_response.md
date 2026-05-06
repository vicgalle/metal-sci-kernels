(1) **Optimization**: Replaced threadgroup memory tiling with direct L1-cache broadcast loads. Removed all `threadgroup_barrier` syncs. Factored the gravitational constant `G` out of the inner loop to save math operations. Heavily unrolled the inner loop 8-way to eliminate branch overhead and expose maximum instruction-level parallelism. 

(2) **Why it improves**: The previous version severely underutilized the GPU for smaller sizes because threadgroup barriers force SIMDgroups to stall, breaking the pipeline. Apple Silicon's L1 cache handles uniform broadcast loads (where all 32 threads in a SIMDgroup read the same `pos_in[j]`) with zero penalty. Combining this with 8-way unrolling and pulling `G` out of the loop pushes the kernel to be entirely compute-bound, maximizing GFLOPS.

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
                       uint i [[thread_position_in_grid]]) {
    
    // Grid matches N exactly, discard out-of-bounds threads immediately.
    // Without threadgroup memory, we have no barriers to deadlock.
    if (i >= N) return;

    float3 ri = pos_in[i].xyz;
    float3 vi = vel_in[i].xyz;
    float eps2 = eps * eps;
    
    float3 sum_a = float3(0.0f);

    uint N8 = N / 8;

    // Process bodies in unrolled chunks of 8 to maximize ILP and pipeline loads
    for (uint j = 0; j < N8; ++j) {
        #pragma unroll
        for (uint k = 0; k < 8; ++k) {
            uint idx = j * 8 + k;
            
            // L1 broadcast load: all threads in SIMDgroup hit the same element
            float4 p = pos_in[idx];
            float m = mass[idx];
            
            float3 d = p.xyz - ri;
            float r2 = dot(d, d) + eps2;
            
            // Fast inverse square root
            float inv_r = rsqrt(r2);
            float inv_r2 = inv_r * inv_r;
            
            // m * (r2)^(-3/2)
            float m_inv_r3 = m * inv_r * inv_r2;
            
            sum_a += d * m_inv_r3;
        }
    }

    // Handle remaining bodies if N is not a multiple of 8
    for (uint j = N8 * 8; j < N; ++j) {
        float4 p = pos_in[j];
        float m = mass[j];
        
        float3 d = p.xyz - ri;
        float r2 = dot(d, d) + eps2;
        
        float inv_r = rsqrt(r2);
        float inv_r2 = inv_r * inv_r;
        float m_inv_r3 = m * inv_r * inv_r2;
        
        sum_a += d * m_inv_r3;
    }

    // Multiply by G once outside the loop to save N instructions
    float3 a = G * sum_a;
    
    // Leapfrog integration
    float3 v_new = vi + a * dt;
    float3 r_new = ri + v_new * dt;

    pos_out[i] = float4(r_new, 0.0f);
    vel_out[i] = float4(v_new, 0.0f);
}
```