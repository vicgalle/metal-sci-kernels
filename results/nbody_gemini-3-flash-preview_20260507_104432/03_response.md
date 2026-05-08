(1) Optimization: I am using threadgroup-memory tiling with a fixed-size `shared_tile` (256 elements) to cooperatively load both body positions and masses. This reduces global memory bandwidth requirements by a factor equal to the tile size. I've also streamlined the inner loop by using `fma` (fused multiply-add) and ensured the compiler can unroll it effectively. (2) Why: N-body simulations are typically instruction-bound due to the $O(N^2)$ complexity. Moving data to threadgroup memory and optimizing the inner acceleration accumulation minimizes latency and maximizes throughput of the floating-point units. The `fma` instruction combined with a single `rsqrt` call per interaction provides the most efficient calculation path on Apple Silicon's execution units.

```metal
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

    // Load current body's position once into registers
    const float3 ri = pos_in[i].xyz;
    float3 acc = float3(0.0f);
    const float eps2 = eps * eps;
    const float Gdt = G * dt;
    
    // Shared memory for cooperative tiling
    // 256 is the standard SIMD/threadgroup size hint for Apple GPUs
    threadgroup float4 shared_tile[256];

    // Iterate through all bodies in the system using tiling
    for (uint j_tile = 0; j_tile < N; j_tile += tpg) {
        // Cooperative load: each thread in the group fetches one body
        const uint j_idx = j_tile + ti;
        if (j_idx < N) {
            float4 p = pos_in[j_idx];
            p.w = mass[j_idx]; // Store mass in the padding of float4
            shared_tile[ti] = p;
        } else {
            shared_tile[ti] = float4(0.0f); // Zero mass means no acceleration contribution
        }

        // Synchronize to ensure the tile is fully loaded
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate acceleration from all bodies in the current tile
        // Unroll the loop to improve pipeline utilization
        #pragma unroll(8)
        for (uint j = 0; j < tpg; ++j) {
            const float4 pj = shared_tile[j];
            const float3 d = pj.xyz - ri;
            const float r2 = dot(d, d) + eps2;
            const float inv_r = rsqrt(r2);
            const float inv_r3 = (inv_r * inv_r) * inv_r;
            // Use Fused Multiply-Add for speed and precision
            acc = fma(d, pj.w * inv_r3, acc);
        }

        // Synchronize before loading the next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Leapfrog Integration:
    // v_new = v_old + a * dt
    // r_new = r_old + v_new * dt
    const float3 vi = vel_in[i].xyz;
    const float3 v_next = fma(acc, Gdt, vi);
    const float3 r_next = fma(v_next, dt, ri);

    // Write results back to global memory
    pos_out[i] = float4(r_next, 0.0f);
    vel_out[i] = float4(v_next, 0.0f);
}
```