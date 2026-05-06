To optimize the baseline all-pairs N-body kernel, we apply **threadgroup memory tiling** to drastically reduce global memory bandwidth. Instead of each thread reading all $N$ positions and masses, threadgroups cooperatively load blocks of elements into fast `threadgroup` memory. 

Key optimizations applied:
1. **Threadgroup Caching**: Active threads load chunks of size `tsize` into shared memory, reducing global memory reads from $O(N^2)$ to $O(N)$.
2. **ILP Unrolling**: The inner force-accumulation loop is manually unrolled by 4, with math operations grouped to maximize Instruction-Level Parallelism (ILP).
3. **Pre-computation**: The gravitational constant `G` is multiplied by `mass` during the shared memory load, saving $N$ multiplications per thread inside the inner loop.
4. **Division Elimination**: The outer tile loop increments `j_start += tsize` directly, avoiding expensive integer division/modulo on the GPU.
5. **Barrier Safety**: Early exits (`if (i >= N) return;`) are removed, ensuring no deadlocks occur at `threadgroup_barrier` if the grid contains partial threadgroups. 

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
                       uint tid [[thread_position_in_threadgroup]],
                       uint tsize [[threads_per_threadgroup]]) {
    
    // Apple Silicon max threadgroup size is 1024
    threadgroup float4 shared_pos[1024];

    // Do not return early to prevent deadlocks at threadgroup_barrier.
    // Use `valid` flag to safely mask outputs and prevent out-of-bounds reads.
    bool valid = i < N;
    
    float3 ri = valid ? pos_in[i].xyz : float3(0.0f);
    float3 vi = valid ? vel_in[i].xyz : float3(0.0f);
    float3 a  = float3(0.0f);
    float eps2 = eps * eps;

    // Iterate over the grid in chunks of `tsize`
    for (uint j_start = 0; j_start < N; j_start += tsize) {
        
        // Cooperatively load tile into threadgroup memory
        uint j = j_start + tid;
        if (j < N) {
            // Pre-multiply mass by G here to save instructions in the hot loop
            shared_pos[tid] = float4(pos_in[j].xyz, G * mass[j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        uint tile_end = min(tsize, N - j_start);
        uint k = 0;
        
        // Unroll heavily for better Instruction-Level Parallelism
        for (; k + 3 < tile_end; k += 4) {
            float4 pj0 = shared_pos[k];
            float4 pj1 = shared_pos[k+1];
            float4 pj2 = shared_pos[k+2];
            float4 pj3 = shared_pos[k+3];
            
            float3 d0 = pj0.xyz - ri;
            float3 d1 = pj1.xyz - ri;
            float3 d2 = pj2.xyz - ri;
            float3 d3 = pj3.xyz - ri;

            float r2_0 = dot(d0, d0) + eps2;
            float r2_1 = dot(d1, d1) + eps2;
            float r2_2 = dot(d2, d2) + eps2;
            float r2_3 = dot(d3, d3) + eps2;

            float inv_r3_0 = rsqrt(r2_0 * r2_0 * r2_0);
            float inv_r3_1 = rsqrt(r2_1 * r2_1 * r2_1);
            float inv_r3_2 = rsqrt(r2_2 * r2_2 * r2_2);
            float inv_r3_3 = rsqrt(r2_3 * r2_3 * r2_3);

            a += pj0.w * d0 * inv_r3_0;
            a += pj1.w * d1 * inv_r3_1;
            a += pj2.w * d2 * inv_r3_2;
            a += pj3.w * d3 * inv_r3_3;
        }
        
        // Handle tail loop for remaining bodies in the tile
        for (; k < tile_end; ++k) {
            float4 pj = shared_pos[k];
            float3 d  = pj.xyz - ri;
            float r2  = dot(d, d) + eps2;
            a += pj.w * d * rsqrt(r2 * r2 * r2);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid) {
        float3 v_new = vi + a * dt;
        float3 r_new = ri + v_new * dt;

        pos_out[i] = float4(r_new, 0.0f);
        vel_out[i] = float4(v_new, 0.0f);
    }
}
```