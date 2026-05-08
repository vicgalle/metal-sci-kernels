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

[[max_total_threads_per_threadgroup(64)]]
kernel void nbody_step(device const float4 *pos_in  [[buffer(0)]],
                       device       float4 *pos_out [[buffer(1)]],
                       device const float4 *vel_in  [[buffer(2)]],
                       device       float4 *vel_out [[buffer(3)]],
                       device const float  *mass    [[buffer(4)]],
                       constant uint        &N      [[buffer(5)]],
                       constant float       &dt     [[buffer(6)]],
                       constant float       &eps    [[buffer(7)]],
                       constant float       &G      [[buffer(8)]],
                       uint i [[thread_position_in_grid]])
{
    if (i >= N) return;
    
    float3 ri = pos_in[i].xyz;
    float3 vi = vel_in[i].xyz;
    float eps2 = eps * eps;
    
    // Split accumulators to hide FMA instruction latency
    float4 ax0 = 0.0f, ay0 = 0.0f, az0 = 0.0f;
    float4 ax1 = 0.0f, ay1 = 0.0f, az1 = 0.0f;
    
    // Cast mass buffer to float4 to vectorize loads
    device const float4* mass4 = (device const float4*)mass;
    
    uint j = 0;
    
    // Unroll by 8, processing 4 interactions per vector
    #pragma unroll(1)
    for (; j + 7 < N; j += 8) {
        float4 p0 = pos_in[j];
        float4 p1 = pos_in[j+1];
        float4 p2 = pos_in[j+2];
        float4 p3 = pos_in[j+3];
        float4 p4 = pos_in[j+4];
        float4 p5 = pos_in[j+5];
        float4 p6 = pos_in[j+6];
        float4 p7 = pos_in[j+7];
        
        float4 m0 = mass4[j >> 2];
        float4 m1 = mass4[(j >> 2) + 1];
        
        // Transpose spatial coordinates into vectorized lanes
        float4 dx0 = float4(p0.x, p1.x, p2.x, p3.x) - ri.x;
        float4 dy0 = float4(p0.y, p1.y, p2.y, p3.y) - ri.y;
        float4 dz0 = float4(p0.z, p1.z, p2.z, p3.z) - ri.z;
        
        float4 dx1 = float4(p4.x, p5.x, p6.x, p7.x) - ri.x;
        float4 dy1 = float4(p4.y, p5.y, p6.y, p7.y) - ri.y;
        float4 dz1 = float4(p4.z, p5.z, p6.z, p7.z) - ri.z;
        
        // Explicit nested fma for r2 calculation
        float4 r2_0 = fma(dx0, dx0, fma(dy0, dy0, fma(dz0, dz0, eps2)));
        float4 r2_1 = fma(dx1, dx1, fma(dy1, dy1, fma(dz1, dz1, eps2)));
        
        float4 inv_r_0 = fast::rsqrt(r2_0);
        float4 inv_r_1 = fast::rsqrt(r2_1);
        
        float4 inv_r3_0 = inv_r_0 * inv_r_0 * inv_r_0;
        float4 inv_r3_1 = inv_r_1 * inv_r_1 * inv_r_1;
        
        float4 f0 = m0 * inv_r3_0;
        float4 f1 = m1 * inv_r3_1;
        
        // Accumulate forces
        ax0 = fma(f0, dx0, ax0);
        ay0 = fma(f0, dy0, ay0);
        az0 = fma(f0, dz0, az0);
        
        ax1 = fma(f1, dx1, ax1);
        ay1 = fma(f1, dy1, ay1);
        az1 = fma(f1, dz1, az1);
    }
    
    // Reduce vector accumulators
    float3 a = float3(ax0.x + ax0.y + ax0.z + ax0.w + ax1.x + ax1.y + ax1.z + ax1.w,
                      ay0.x + ay0.y + ay0.z + ay0.w + ay1.x + ay1.y + ay1.z + ay1.w,
                      az0.x + az0.y + az0.z + az0.w + az1.x + az1.y + az1.z + az1.w);
                      
    // Remainder loop
    #pragma unroll(1)
    for (; j < N; ++j) {
        float3 p = pos_in[j].xyz;
        float3 d = p - ri;
        float r2 = fma(d.x, d.x, fma(d.y, d.y, fma(d.z, d.z, eps2)));
        float inv_r = fast::rsqrt(r2);
        float f = mass[j] * inv_r * inv_r * inv_r;
        a += f * d;
    }
    
    a *= G;
    float3 v_new = vi + a * dt;
    float3 r_new = ri + v_new * dt;
    
    pos_out[i] = float4(r_new, 0.0f);
    vel_out[i] = float4(v_new, 0.0f);
}
```

Result of previous attempt:
            256_10: correct, 0.49 ms, 27.0 GFLOPS (0.6% of 4500 GFLOPS)
           1024_10: correct, 1.70 ms, 123.1 GFLOPS (2.7% of 4500 GFLOPS)
           2048_10: correct, 2.43 ms, 344.7 GFLOPS (7.7% of 4500 GFLOPS)
  score (gmean of fraction): 0.0233

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

- iter  0: compile=OK | correct=True | score=0.02043019588483495
- iter  1: compile=OK | correct=True | score=0.03206268136544089
- iter  2: compile=OK | correct=True | score=0.033807647331100914
- iter  3: compile=FAIL | correct=False | score=N/A
- iter  4: compile=OK | correct=True | score=0.0408083015870597
- iter  5: compile=OK | correct=True | score=0.023250149123621833

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
