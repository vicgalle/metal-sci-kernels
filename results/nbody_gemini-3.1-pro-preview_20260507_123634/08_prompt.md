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
    
    if (i < N) {
        ri = pos_in[i].xyz;
        vi = vel_in[i].xyz;
    }
    
    float eps2 = eps * eps;
    float4 eps2_v = float4(eps2);
    
    // 4 independent sets of accumulators to hide loop-carried dependency latency
    float ax0 = 0.0f, ay0 = 0.0f, az0 = 0.0f;
    float ax1 = 0.0f, ay1 = 0.0f, az1 = 0.0f;
    float ax2 = 0.0f, ay2 = 0.0f, az2 = 0.0f;
    float ax3 = 0.0f, ay3 = 0.0f, az3 = 0.0f;
    
    // Structure-of-Arrays layout statically allocated to Metal's max threadgroup size (16 KB total)
    threadgroup float local_x[1024] __attribute__((aligned(16)));
    threadgroup float local_y[1024] __attribute__((aligned(16)));
    threadgroup float local_z[1024] __attribute__((aligned(16)));
    threadgroup float local_m[1024] __attribute__((aligned(16)));
    
    for (uint t = 0; t < N; t += tg_size) {
        uint limit = min(tg_size, N - t);
        
        // Cooperative load converting from AoS (global) to SoA (threadgroup)
        if (ti < limit) {
            float4 p = pos_in[t + ti];
            local_x[ti] = p.x;
            local_y[ti] = p.y;
            local_z[ti] = p.z;
            local_m[ti] = mass[t + ti];
        }
        
        // Wait for all tile data to be visible
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        uint k = 0;
        // Unroll by 16: processes 16 particles across 4 block iterations per loop step
        for (; k + 15 < limit; k += 16) {
            // Block 0
            float4 px0 = *(threadgroup float4*)(&local_x[k]);
            float4 py0 = *(threadgroup float4*)(&local_y[k]);
            float4 pz0 = *(threadgroup float4*)(&local_z[k]);
            float4 m0  = *(threadgroup float4*)(&local_m[k]);
            
            // Block 1
            float4 px1 = *(threadgroup float4*)(&local_x[k+4]);
            float4 py1 = *(threadgroup float4*)(&local_y[k+4]);
            float4 pz1 = *(threadgroup float4*)(&local_z[k+4]);
            float4 m1  = *(threadgroup float4*)(&local_m[k+4]);

            // Block 2
            float4 px2 = *(threadgroup float4*)(&local_x[k+8]);
            float4 py2 = *(threadgroup float4*)(&local_y[k+8]);
            float4 pz2 = *(threadgroup float4*)(&local_z[k+8]);
            float4 m2  = *(threadgroup float4*)(&local_m[k+8]);
            
            // Block 3
            float4 px3 = *(threadgroup float4*)(&local_x[k+12]);
            float4 py3 = *(threadgroup float4*)(&local_y[k+12]);
            float4 pz3 = *(threadgroup float4*)(&local_z[k+12]);
            float4 m3  = *(threadgroup float4*)(&local_m[k+12]);
            
            float4 dx0 = px0 - ri.x;
            float4 dy0 = py0 - ri.y;
            float4 dz0 = pz0 - ri.z;
            
            float4 dx1 = px1 - ri.x;
            float4 dy1 = py1 - ri.y;
            float4 dz1 = pz1 - ri.z;

            float4 dx2 = px2 - ri.x;
            float4 dy2 = py2 - ri.y;
            float4 dz2 = pz2 - ri.z;
            
            float4 dx3 = px3 - ri.x;
            float4 dy3 = py3 - ri.y;
            float4 dz3 = pz3 - ri.z;
            
            float4 r2_0 = fma(dz0, dz0, fma(dy0, dy0, fma(dx0, dx0, eps2_v)));
            float4 r2_1 = fma(dz1, dz1, fma(dy1, dy1, fma(dx1, dx1, eps2_v)));
            float4 r2_2 = fma(dz2, dz2, fma(dy2, dy2, fma(dx2, dx2, eps2_v)));
            float4 r2_3 = fma(dz3, dz3, fma(dy3, dy3, fma(dx3, dx3, eps2_v)));
            
            float4 inv_r_0 = fast::rsqrt(r2_0);
            float4 inv_r_1 = fast::rsqrt(r2_1);
            float4 inv_r_2 = fast::rsqrt(r2_2);
            float4 inv_r_3 = fast::rsqrt(r2_3);
            
            float4 inv_r3_0 = inv_r_0 * (inv_r_0 * inv_r_0);
            float4 inv_r3_1 = inv_r_1 * (inv_r_1 * inv_r_1);
            float4 inv_r3_2 = inv_r_2 * (inv_r_2 * inv_r_2);
            float4 inv_r3_3 = inv_r_3 * (inv_r_3 * inv_r_3);
            
            float4 f0 = m0 * inv_r3_0;
            float4 f1 = m1 * inv_r3_1;
            float4 f2 = m2 * inv_r3_2;
            float4 f3 = m3 * inv_r3_3;
            
            // Apple Silicon's 1-cycle dot matches mathematical component summation across 4 particles
            ax0 += dot(f0, dx0);
            ay0 += dot(f0, dy0);
            az0 += dot(f0, dz0);
            
            ax1 += dot(f1, dx1);
            ay1 += dot(f1, dy1);
            az1 += dot(f1, dz1);

            ax2 += dot(f2, dx2);
            ay2 += dot(f2, dy2);
            az2 += dot(f2, dz2);
            
            ax3 += dot(f3, dx3);
            ay3 += dot(f3, dy3);
            az3 += dot(f3, dz3);
        }
        
        // Remainder loop
        for (; k < limit; ++k) {
            float dx = local_x[k] - ri.x;
            float dy = local_y[k] - ri.y;
            float dz = local_z[k] - ri.z;
            
            float r2 = fma(dz, dz, fma(dy, dy, fma(dx, dx, eps2)));
            float inv_r = fast::rsqrt(r2);
            float f = local_m[k] * (inv_r * inv_r * inv_r);
            
            ax0 += f * dx;
            ay0 += f * dy;
            az0 += f * dz;
        }
        
        // Guard before overwriting tile arrays on the next iteration
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Combine and apply updates
    if (i < N) {
        float3 a = float3(ax0 + ax1 + ax2 + ax3,
                          ay0 + ay1 + ay2 + ay3,
                          az0 + az1 + az2 + az3) * G;
        
        float3 v_new = vi + a * dt;
        float3 r_new = ri + v_new * dt;
        
        pos_out[i] = float4(r_new, 0.0f);
        vel_out[i] = float4(v_new, 0.0f);
    }
}
```

Result of previous attempt:
            256_10: correct, 1.48 ms, 8.9 GFLOPS (0.2% of 4500 GFLOPS)
           1024_10: correct, 5.76 ms, 36.4 GFLOPS (0.8% of 4500 GFLOPS)
           2048_10: correct, 6.96 ms, 120.6 GFLOPS (2.7% of 4500 GFLOPS)
  score (gmean of fraction): 0.0075

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
- iter  6: compile=OK | correct=True | score=0.0309432526329866
- iter  7: compile=OK | correct=True | score=0.007527450619508877

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
