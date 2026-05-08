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
                       constant uint        &N_ref  [[buffer(5)]],
                       constant float       &dt_ref [[buffer(6)]],
                       constant float       &eps_ref[[buffer(7)]],
                       constant float       &G_ref  [[buffer(8)]],
                       uint i [[thread_position_in_grid]],
                       uint ti [[thread_position_in_threadgroup]],
                       uint tg_size [[threads_per_threadgroup]]) [[max_total_threads_per_threadgroup(256)]]
{
    uint N = N_ref;
    float dt = dt_ref;
    float eps2 = eps_ref * eps_ref;
    float G = G_ref;
    
    // Mask out-of-bounds reads safely while keeping threads active for barriers
    float4 p_i = (i < N) ? pos_in[i] : float4(0.0f);
    float3 ri = p_i.xyz;
    float3 vi = (i < N) ? vel_in[i].xyz : float3(0.0f);
    
    // Split accumulators hide FMA latencies
    float3 a_0 = float3(0.0f);
    float3 a_1 = float3(0.0f);
    
    threadgroup float4 local_pos[256];
    threadgroup float  local_mass[256];
    
    for (uint t = 0; t < N; t += tg_size) {
        uint limit = min(tg_size, N - t);
        
        // Cooperative loading into threadgroup memory
        if (ti < limit) {
            local_pos[ti] = pos_in[t + ti];
            local_mass[ti] = mass[t + ti];
        }
        
        // Synchronize all active threads
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        uint k = 0;
        // Unroll by 8 to maximize ILP
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
            
            // Scalar fast rsqrt maps efficiently without pack/unpack overhead
            float inv_r_0 = fast::rsqrt(r2_0);
            float inv_r_1 = fast::rsqrt(r2_1);
            float inv_r_2 = fast::rsqrt(r2_2);
            float inv_r_3 = fast::rsqrt(r2_3);
            float inv_r_4 = fast::rsqrt(r2_4);
            float inv_r_5 = fast::rsqrt(r2_5);
            float inv_r_6 = fast::rsqrt(r2_6);
            float inv_r_7 = fast::rsqrt(r2_7);
            
            float f0 = local_mass[k]   * (inv_r_0 * inv_r_0 * inv_r_0);
            float f1 = local_mass[k+1] * (inv_r_1 * inv_r_1 * inv_r_1);
            float f2 = local_mass[k+2] * (inv_r_2 * inv_r_2 * inv_r_2);
            float f3 = local_mass[k+3] * (inv_r_3 * inv_r_3 * inv_r_3);
            float f4 = local_mass[k+4] * (inv_r_4 * inv_r_4 * inv_r_4);
            float f5 = local_mass[k+5] * (inv_r_5 * inv_r_5 * inv_r_5);
            float f6 = local_mass[k+6] * (inv_r_6 * inv_r_6 * inv_r_6);
            float f7 = local_mass[k+7] * (inv_r_7 * inv_r_7 * inv_r_7);
            
            a_0 += f0 * d0;
            a_1 += f1 * d1;
            a_0 += f2 * d2;
            a_1 += f3 * d3;
            a_0 += f4 * d4;
            a_1 += f5 * d5;
            a_0 += f6 * d6;
            a_1 += f7 * d7;
        }
        
        // Remainder loop
        for (; k < limit; ++k) {
            float3 d = local_pos[k].xyz - ri;
            float r2 = dot(d, d) + eps2;
            float inv_r = fast::rsqrt(r2);
            float f = local_mass[k] * (inv_r * inv_r * inv_r);
            a_0 += f * d;
        }
        
        // Synchronize before loading the next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Guard the final write to memory so only valid bodies produce output
    if (i < N) {
        float3 a = (a_0 + a_1) * G;
        float3 v_new = vi + a * dt;
        float3 r_new = ri + v_new * dt;
        
        pos_out[i] = float4(r_new, 0.0f);
        vel_out[i] = float4(v_new, 0.0f);
    }
}
```

Result of previous attempt:
  COMPILE FAILED: Error Domain=MTLLibraryErrorDomain Code=3 "program_source:15:68: error: 'max_total_threads_per_threadgroup' attribute cannot be applied to types
                       uint tg_size [[threads_per_threadgroup]]) [[max_total_threads_per_threadgroup(256)]]
                                                                   ^
" UserInfo={NSLocalizedDescription=program_source:15:68: error: 'max_total_threads_per_threadgroup' attribute cannot be applied to types
                       uint tg_size [[threads_per_threadgroup]]) [[max_total_threads_per_threadgroup(256)]]
                                                                   ^
}

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

kernel void nbody_step(device const float4 *pos_in  [[buffer(0)]],
                       device       float4 *pos_out [[buffer(1)]],
                       device const float4 *vel_in  [[buffer(2)]],
                       device       float4 *vel_out [[buffer(3)]],
                       device const float  *mass    [[buffer(4)]],
                       constant uint        &N_ref  [[buffer(5)]],
                       constant float       &dt_ref [[buffer(6)]],
                       constant float       &eps_ref[[buffer(7)]],
                       constant float       &G_ref  [[buffer(8)]],
                       uint i [[thread_position_in_grid]])
{
    uint N = N_ref;
    if (i >= N) return;
    
    float dt = dt_ref;
    float eps = eps_ref;
    float G = G_ref;
    
    float4 p_i = pos_in[i];
    float3 ri = p_i.xyz;
    float3 vi = vel_in[i].xyz;
    
    float3 a = float3(0.0f);
    float eps2 = eps * eps;
    
    uint j = 0;
    // Unroll by 8 to maximize Instruction-Level Parallelism (ILP)
    for (; j + 7 < N; j += 8) {
        float4 p0 = pos_in[j];
        float4 p1 = pos_in[j+1];
        float4 p2 = pos_in[j+2];
        float4 p3 = pos_in[j+3];
        float4 p4 = pos_in[j+4];
        float4 p5 = pos_in[j+5];
        float4 p6 = pos_in[j+6];
        float4 p7 = pos_in[j+7];
        
        // Coalesced loads for the scalar mass array
        float4 m_vec0 = *(device const float4*)(&mass[j]);
        float4 m_vec1 = *(device const float4*)(&mass[j+4]);
        
        float3 d0 = p0.xyz - ri;
        float3 d1 = p1.xyz - ri;
        float3 d2 = p2.xyz - ri;
        float3 d3 = p3.xyz - ri;
        float3 d4 = p4.xyz - ri;
        float3 d5 = p5.xyz - ri;
        float3 d6 = p6.xyz - ri;
        float3 d7 = p7.xyz - ri;
        
        float r2_0 = dot(d0, d0) + eps2;
        float r2_1 = dot(d1, d1) + eps2;
        float r2_2 = dot(d2, d2) + eps2;
        float r2_3 = dot(d3, d3) + eps2;
        float r2_4 = dot(d4, d4) + eps2;
        float r2_5 = dot(d5, d5) + eps2;
        float r2_6 = dot(d6, d6) + eps2;
        float r2_7 = dot(d7, d7) + eps2;
        
        // Pack into float4 vectors for fast SIMD reciprocal square root
        float4 r2_vec0 = float4(r2_0, r2_1, r2_2, r2_3);
        float4 r2_vec1 = float4(r2_4, r2_5, r2_6, r2_7);
        
        float4 inv_r_vec0 = fast::rsqrt(r2_vec0);
        float4 inv_r_vec1 = fast::rsqrt(r2_vec1);
        
        float4 inv_r3_vec0 = inv_r_vec0 * inv_r_vec0 * inv_r_vec0;
        float4 inv_r3_vec1 = inv_r_vec1 * inv_r_vec1 * inv_r_vec1;
        
        float4 f_vec0 = m_vec0 * inv_r3_vec0;
        float4 f_vec1 = m_vec1 * inv_r3_vec1;
        
        a = fma(f_vec0.x, d0, a);
        a = fma(f_vec0.y, d1, a);
        a = fma(f_vec0.z, d2, a);
        a = fma(f_vec0.w, d3, a);
        a = fma(f_vec1.x, d4, a);
        a = fma(f_vec1.y, d5, a);
        a = fma(f_vec1.z, d6, a);
        a = fma(f_vec1.w, d7, a);
    }
    
    // Process remaining elements if N is not a multiple of 8
    for (; j < N; ++j) {
        float4 p = pos_in[j];
        float m = mass[j];
        float3 d = p.xyz - ri;
        float r2 = dot(d, d) + eps2;
        float inv_r = fast::rsqrt(r2);
        float inv_r3 = inv_r * inv_r * inv_r;
        float f = m * inv_r3;
        a = fma(f, d, a);
    }
    
    // Apply the factored gravitational constant
    a *= G;
    
    float3 v_new = vi + a * dt;
    float3 r_new = ri + v_new * dt;
    
    pos_out[i] = float4(r_new, 0.0f);
    vel_out[i] = float4(v_new, 0.0f);
}
```

Incumbent result:
            256_10: correct, 0.38 ms, 34.4 GFLOPS (0.8% of 4500 GFLOPS)
           1024_10: correct, 1.35 ms, 154.8 GFLOPS (3.4% of 4500 GFLOPS)
           2048_10: correct, 1.27 ms, 661.7 GFLOPS (14.7% of 4500 GFLOPS)
  score (gmean of fraction): 0.0338

## History

- iter  0: compile=OK | correct=True | score=0.02043019588483495
- iter  1: compile=OK | correct=True | score=0.03206268136544089
- iter  2: compile=OK | correct=True | score=0.033807647331100914
- iter  3: compile=FAIL | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
