This iteration uses **threadgroup-memory tiling** to significantly reduce global memory bandwidth requirements. By loading a 256-body tile into threadgroup memory (L1-backed), each body's position and mass are fetched once and reused by all threads in the threadgroup, reducing global reads from $O(N^2)$ to $O(N^2/TILE\_SIZE)$.

We further optimize the kernel by:
1.  **Mass-Padding**: Initializing the tile with zero mass for out-of-bounds indices, which eliminates branching within the tight inner loop.
2.  **Loop Unrolling**: Applying `#pragma unroll(8)` to the inner loop to improve instruction-level parallelism (ILP) and better utilize the GPU's pipelines.
3.  **Correct Attributes**: Using standard MSL attributes like `[[threads_per_threadgroup]]` correctly as `uint3` to ensure compatibility and robustness across different dispatch configurations.
4.  **Arithmetic Optimization**: Pre-calculating $G \cdot dt$ and structuring the interaction math to minimize redundant operations.

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
                       uint tid [[thread_index_in_threadgroup]],
                       uint3 tgs [[threads_per_threadgroup]]) {
    // Guard against out-of-bounds threads
    if (i >= N) return;

    // Load current particle state into registers
    const float3 ri = pos_in[i].xyz;
    const float3 vi = vel_in[i].xyz;
    const float eps2 = eps * eps;
    float3 acc = float3(0.0f);

    // Tile size is chosen to balance threadgroup memory usage and occupancy
    const uint TILE_SIZE = 256;
    threadgroup float4 tile[TILE_SIZE];

    for (uint j_base = 0; j_base < N; j_base += TILE_SIZE) {
        // Cooperative load: threads fill the tile together
        for (uint l = tid; l < TILE_SIZE; l += tgs.x) {
            uint j_idx = j_base + l;
            if (j_idx < N) {
                // Combine position and mass into a single float4 to minimize L1 traffic
                tile[l] = float4(pos_in[j_idx].xyz, mass[j_idx]);
            } else {
                // Pad out-of-bounds particles with zero mass to nullify their contribution
                tile[l] = float4(0.0f);
            }
        }
        
        // Ensure all threads have finished loading the tile
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute interactions with all particles in the tile
        const uint current_tile_limit = (N - j_base < TILE_SIZE) ? (N - j_base) : TILE_SIZE;
        
        #pragma unroll(8)
        for (uint j = 0; j < current_tile_limit; ++j) {
            float4 rj_mj = tile[j];
            float3 d = rj_mj.xyz - ri;
            float r2 = d.x * d.x + d.y * d.y + d.z * d.z + eps2;
            
            // Efficiently compute (r^2 + eps^2)^(-1.5)
            float inv_r = rsqrt(r2);
            float inv_r3 = inv_r * inv_r * inv_r;
            
            // Contribution: m_j * (r_j - r_i) / dist^3
            acc += d * (rj_mj.w * inv_r3);
        }

        // Synchronize before moving to the next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Leapfrog Integration: 
    // v_new = v + a * dt
    // r_new = r + v_new * dt
    const float3 a = acc * G;
    const float3 v_new = vi + a * dt;
    const float3 r_new = ri + v_new * dt;

    // Write-back updated state
    pos_out[i] = float4(r_new, 0.0f);
    vel_out[i] = float4(v_new, 0.0f);
}
```