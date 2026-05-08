## Task: nbody

All-pairs gravitational N-body with leapfrog integration. For each body i:
  a_i = G * sum_{j} m_j (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)
  v_new = v + a * dt
  r_new = r + v_new * dt
Self-interaction is masked by the softening epsilon (no special case needed). Positions/velocities are packed as float4 with .xyz holding the data and .w padding. Masses are float[N].

## Required kernel signature(s)

```
kernel void nbody_step(device const float4 *pos_in  [[buffer(0)]],
                       device       float4 *pos_out [[buffer(1)]],
                       device const float4 *vel_in  [[buffer(2)]],
                       device       float4 *vel_out [[buffer(3)]],
                       device const float  *mass    [[buffer(4)]],
                       constant uint        &N      [[buffer(5)]],
                       constant float       &dt     [[buffer(6)]],
                       constant float       &eps    [[buffer(7)]],
                       constant float       &G      [[buffer(8)]],
                       uint i [[thread_position_in_grid]]);

Threads are dispatched 1-D, one per body — guard with `if (i >= N) return;`. Each thread MUST update exactly one body; the host will not shrink the dispatch if you process multiple bodies per thread, so extra threads just idle. The host ping-pongs (pos_in, pos_out) and (vel_in, vel_out) buffer pairs each step.
```

## Your previous attempt

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
                       uint ti [[thread_position_in_threadgroup]],
                       uint tg_size [[threads_per_threadgroup]]) 
{
    float3 ri = float3(0.0f);
    float3 vi = float3(0.0f);
    
    if (i < N) {
        ri = pos_in[i].xyz;
        vi = vel_in[i].xyz;
    }
    
    float3 a = float3(0.0f);
    float eps2 = eps * eps;
    
    constexpr uint BLOCK_SIZE = 256;
    threadgroup float4 local_pos_mass[BLOCK_SIZE];
    
    for (uint t = 0; t < N; t += BLOCK_SIZE) {
        // Cooperatively load a block of positions and masses into threadgroup memory
        for (uint idx = ti; idx < BLOCK_SIZE; idx += tg_size) {
            uint j = t + idx;
            if (j < N) {
                float4 p = pos_in[j];
                // Pack pre-multiplied G * mass into the unused .w padding component
                p.w = G * mass[j];
                local_pos_mass[idx] = p;
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        uint num_elements = min(BLOCK_SIZE, N - t);
        
        if (i < N) {
            uint k = 0;
            // Unroll loop by 4 to expose instruction-level parallelism (ILP)
            for (; k + 3 < num_elements; k += 4) {
                float4 p0 = local_pos_mass[k];
                float4 p1 = local_pos_mass[k+1];
                float4 p2 = local_pos_mass[k+2];
                float4 p3 = local_pos_mass[k+3];
                
                float3 d0 = p0.xyz - ri;
                float3 d1 = p1.xyz - ri;
                float3 d2 = p2.xyz - ri;
                float3 d3 = p3.xyz - ri;
                
                float r2_0 = dot(d0, d0) + eps2;
                float r2_1 = dot(d1, d1) + eps2;
                float r2_2 = dot(d2, d2) + eps2;
                float r2_3 = dot(d3, d3) + eps2;
                
                float inv_r3_0 = rsqrt(r2_0 * r2_0 * r2_0);
                float inv_r3_1 = rsqrt(r2_1 * r2_1 * r2_1);
                float inv_r3_2 = rsqrt(r2_2 * r2_2 * r2_2);
                float inv_r3_3 = rsqrt(r2_3 * r2_3 * r2_3);
                
                a += p0.w * d0 * inv_r3_0;
                a += p1.w * d1 * inv_r3_1;
                a += p2.w * d2 * inv_r3_2;
                a += p3.w * d3 * inv_r3_3;
            }
            
            // Handle remainder elements
            for (; k < num_elements; ++k) {
                float4 p = local_pos_mass[k];
                float3 d = p.xyz - ri;
                float r2 = dot(d, d) + eps2;
                float inv_r3 = rsqrt(r2 * r2 * r2);
                a += p.w * d * inv_r3;
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (i < N) {
        float3 v_new = vi + a * dt;
        float3 r_new = ri + v_new * dt;
        
        pos_out[i] = float4(r_new, 0.0f);
        vel_out[i] = float4(v_new, 0.0f);
    }
}
```

Result of previous attempt:
            256_10: correct, 0.39 ms, 33.4 GFLOPS (0.7% of 4500 GFLOPS)
           1024_10: correct, 1.40 ms, 149.9 GFLOPS (3.3% of 4500 GFLOPS)
           2048_10: correct, 1.40 ms, 600.7 GFLOPS (13.3% of 4500 GFLOPS)
  score (gmean of fraction): 0.0321

## History

- iter  0: compile=OK | correct=True | score=0.02043019588483495
- iter  1: compile=OK | correct=True | score=0.03206268136544089

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
