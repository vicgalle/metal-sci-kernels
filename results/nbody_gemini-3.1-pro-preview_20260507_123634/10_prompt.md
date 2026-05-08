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

[[max_total_threads_per_threadgroup(1024)]]
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
    
    // Mask out-of-bounds reads safely while keeping threads active for barriers
    if (i < N) {
        ri = pos_in[i].xyz;
        vi = vel_in[i].xyz;
    }
    
    float eps2 = eps * eps;
    float4 eps2_v = float4(eps2);
    
    // 4 accumulators to completely hide FMA instruction latency
    float3 a0 = float3(0.0f);
    float3 a1 = float3(0.0f);
    float3 a2 = float3(0.0f);
    float3 a3 = float3(0.0f);
    
    // Single packed threadgroup array for position (xyz) and mass (w)
    threadgroup float4 local_pos[1024];
    
    for (uint t = 0; t < N; t += tg_size) {
        uint limit = min(tg_size, N - t);
        
        // Cooperative load: seamlessly coalesced 128-byte/32-byte global reads
        if (ti < limit) {
            float4 p = pos_in[t + ti];
            p.w = mass[t + ti];
            local_pos[ti] = p;
        }
        
        // Ensure tile is fully loaded before computation begins
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        uint k = 0;
        // Unroll by 8, but split into 4-element scopes to aggressively reuse registers
        #pragma unroll(1)
        for (; k + 7 < limit; k += 8) {
            {
                float4 p0 = local_pos[k];
                float4 p1 = local_pos[k+1];
                float4 p2 = local_pos[k+2];
                float4 p3 = local_pos[k+3];
                
                float3 d0 = p0.xyz - ri;
                float3 d1 = p1.xyz - ri;
                float3 d2 = p2.xyz - ri;
                float3 d3 = p3.xyz - ri;
                
                // Vectorize purely independent math to leverage 128-bit execution pathways
                float4 r2 = float4(dot(d0, d0), dot(d1, d1), dot(d2, d2), dot(d3, d3)) + eps2_v;
                float4 inv_r = fast::rsqrt(r2);
                float4 inv_r3 = inv_r * inv_r * inv_r;
                float4 f = float4(p0.w, p1.w, p2.w, p3.w) * inv_r3;
                
                a0 = fma(f.x, d0, a0);
                a1 = fma(f.y, d1, a1);
                a2 = fma(f.z, d2, a2);
                a3 = fma(f.w, d3, a3);
            }
            {
                // Identical scope forces the allocator to reuse registers from the previous block
                float4 p0 = local_pos[k+4];
                float4 p1 = local_pos[k+5];
                float4 p2 = local_pos[k+6];
                float4 p3 = local_pos[k+7];
                
                float3 d0 = p0.xyz - ri;
                float3 d1 = p1.xyz - ri;
                float3 d2 = p2.xyz - ri;
                float3 d3 = p3.xyz - ri;
                
                float4 r2 = float4(dot(d0, d0), dot(d1, d1), dot(d2, d2), dot(d3, d3)) + eps2_v;
                float4 inv_r = fast::rsqrt(r2);
                float4 inv_r3 = inv_r * inv_r * inv_r;
                float4 f = float4(p0.w, p1.w, p2.w, p3.w) * inv_r3;
                
                a0 = fma(f.x, d0, a0);
                a1 = fma(f.y, d1, a1);
                a2 = fma(f.z, d2, a2);
                a3 = fma(f.w, d3, a3);
            }
        }
        
        // Small tail remainder
        for (; k < limit; ++k) {
            float4 p = local_pos[k];
            float3 d = p.xyz - ri;
            float r2 = dot(d, d) + eps2;
            float inv_r = fast::rsqrt(r2);
            float f = p.w * (inv_r * inv_r * inv_r);
            a0 = fma(f, d, a0);
        }
        
        // Ensure all threads finish computation before the next cooperative load overwrites
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Guard the final integration step; early returns were avoided to prevent barrier deadlocks
    if (i < N) {
        float3 a = (a0 + a1 + a2 + a3) * G;
        float3 v_new = vi + a * dt;
        float3 r_new = ri + v_new * dt;
        
        pos_out[i] = float4(r_new, 0.0f);
        vel_out[i] = float4(v_new, 0.0f);
    }
}
```

Result of previous attempt:
            256_10: correct, 0.90 ms, 14.5 GFLOPS (0.3% of 4500 GFLOPS)
           1024_10: correct, 1.56 ms, 134.7 GFLOPS (3.0% of 4500 GFLOPS)
           2048_10: correct, 2.00 ms, 419.9 GFLOPS (9.3% of 4500 GFLOPS)
  score (gmean of fraction): 0.0208

## Current best (incumbent)

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
    
    // Mask out-of-bounds reads safely while keeping threads active for barriers
    if (i < N) {
        ri = pos_in[i].xyz;
        vi = vel_in[i].xyz;
    }
    
    float eps2 = eps * eps;
    
    // Split accumulators to hide FMA instruction latency
    float3 a0 = float3(0.0f);
    float3 a1 = float3(0.0f);
    
    // Statically allocate enough for the maximum Metal threadgroup size
    threadgroup float4 local_pos[1024];
    threadgroup float  local_mass[1024] __attribute__((aligned(16)));
    
    for (uint t = 0; t < N; t += tg_size) {
        uint limit = min(tg_size, N - t);
        
        // Cooperative load into threadgroup memory
        if (ti < limit) {
            local_pos[ti] = pos_in[t + ti];
            local_mass[ti] = mass[t + ti];
        }
        
        // Ensure tile is fully loaded before computation begins
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        uint k = 0;
        // Unroll by 8 to maximize ILP and explicitly vectorize fast::rsqrt
        for (; k + 7 < limit; k += 8) {
            float3 d0 = local_pos[k].xyz - ri;
            float3 d1 = local_pos[k+1].xyz - ri;
            float3 d2 = local_pos[k+2].xyz - ri;
            float3 d3 = local_pos[k+3].xyz - ri;
            float3 d4 = local_pos[k+4].xyz - ri;
            float3 d5 = local_pos[k+5].xyz - ri;
            float3 d6 = local_pos[k+6].xyz - ri;
            float3 d7 = local_pos[k+7].xyz - ri;
            
            float r2_0 = dot(d0, d0) + eps2;
            float r2_1 = dot(d1, d1) + eps2;
            float r2_2 = dot(d2, d2) + eps2;
            float r2_3 = dot(d3, d3) + eps2;
            float r2_4 = dot(d4, d4) + eps2;
            float r2_5 = dot(d5, d5) + eps2;
            float r2_6 = dot(d6, d6) + eps2;
            float r2_7 = dot(d7, d7) + eps2;
            
            float4 r2_v0 = float4(r2_0, r2_1, r2_2, r2_3);
            float4 r2_v1 = float4(r2_4, r2_5, r2_6, r2_7);
            
            float4 inv_r_v0 = fast::rsqrt(r2_v0);
            float4 inv_r_v1 = fast::rsqrt(r2_v1);
            
            float4 inv_r3_v0 = inv_r_v0 * inv_r_v0 * inv_r_v0;
            float4 inv_r3_v1 = inv_r_v1 * inv_r_v1 * inv_r_v1;
            
            // Cast to vectorize memory reads inside the 16-byte aligned threadgroup block
            float4 m_v0 = *(threadgroup float4*)(&local_mass[k]);
            float4 m_v1 = *(threadgroup float4*)(&local_mass[k+4]);
            
            float4 f_v0 = m_v0 * inv_r3_v0;
            float4 f_v1 = m_v1 * inv_r3_v1;
            
            a0 = fma(f_v0.x, d0, a0);
            a1 = fma(f_v0.y, d1, a1);
            a0 = fma(f_v0.z, d2, a0);
            a1 = fma(f_v0.w, d3, a1);
            a0 = fma(f_v1.x, d4, a0);
            a1 = fma(f_v1.y, d5, a1);
            a0 = fma(f_v1.z, d6, a0);
            a1 = fma(f_v1.w, d7, a1);
        }
        
        // Remainder loop processing within the local tile
        for (; k < limit; ++k) {
            float3 d = local_pos[k].xyz - ri;
            float r2 = dot(d, d) + eps2;
            float inv_r = fast::rsqrt(r2);
            float f = local_mass[k] * (inv_r * inv_r * inv_r);
            a0 = fma(f, d, a0);
        }
        
        // Ensure all computations are done before overwriting the next tile memory
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Guard the final write to memory so only explicitly valid bodies produce output
    if (i < N) {
        float3 a = (a0 + a1) * G;
        float3 v_new = vi + a * dt;
        float3 r_new = ri + v_new * dt;
        
        pos_out[i] = float4(r_new, 0.0f);
        vel_out[i] = float4(v_new, 0.0f);
    }
}
```

Incumbent result:
            256_10: correct, 0.39 ms, 33.2 GFLOPS (0.7% of 4500 GFLOPS)
           1024_10: correct, 0.72 ms, 289.5 GFLOPS (6.4% of 4500 GFLOPS)
           2048_10: correct, 1.30 ms, 643.9 GFLOPS (14.3% of 4500 GFLOPS)
  score (gmean of fraction): 0.0408

## History

- iter  2: compile=OK | correct=True | score=0.033807647331100914
- iter  3: compile=FAIL | correct=False | score=N/A
- iter  4: compile=OK | correct=True | score=0.0408083015870597
- iter  5: compile=OK | correct=True | score=0.023250149123621833
- iter  6: compile=OK | correct=True | score=0.0309432526329866
- iter  7: compile=OK | correct=True | score=0.007527450619508877
- iter  8: compile=OK | correct=True | score=0.014540294322401465
- iter  9: compile=OK | correct=True | score=0.020810488822513084

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
