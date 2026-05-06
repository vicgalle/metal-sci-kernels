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

Threads are dispatched 1-D over N. Guard with `if (i >= N) return;`. The host ping-pongs (pos_in, pos_out) and (vel_in, vel_out) buffer pairs each step.
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
                       uint tid [[thread_position_in_threadgroup]],
                       uint tsize [[threads_per_threadgroup]]) [[max_total_threads_per_threadgroup(1024)]] {
    
    threadgroup float4 shared_pos[1024];

    // Do not return early; keep SIMD-groups intact for the barrier to prevent deadlocks.
    bool valid = i < N;
    
    float3 ri = valid ? pos_in[i].xyz : float3(0.0f);
    float3 vi = valid ? vel_in[i].xyz : float3(0.0f);
    float3 a  = float3(0.0f);
    float eps2 = eps * eps;

    for (uint j_start = 0; j_start < N; j_start += tsize) {
        
        uint j = j_start + tid;
        if (j < N) {
            // Premultiplying mass by G directly on load saves instructions later
            shared_pos[tid] = float4(pos_in[j].xyz, G * mass[j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        uint tile_end = min(tsize, N - j_start);
        uint k = 0;
        
        // 8x unrolled loop, meticulously grouped to expose massive ILP
        for (; k + 7 < tile_end; k += 8) {
            float4 pj0 = shared_pos[k];
            float4 pj1 = shared_pos[k+1];
            float4 pj2 = shared_pos[k+2];
            float4 pj3 = shared_pos[k+3];
            float4 pj4 = shared_pos[k+4];
            float4 pj5 = shared_pos[k+5];
            float4 pj6 = shared_pos[k+6];
            float4 pj7 = shared_pos[k+7];
            
            float3 d0 = pj0.xyz - ri;
            float3 d1 = pj1.xyz - ri;
            float3 d2 = pj2.xyz - ri;
            float3 d3 = pj3.xyz - ri;
            float3 d4 = pj4.xyz - ri;
            float3 d5 = pj5.xyz - ri;
            float3 d6 = pj6.xyz - ri;
            float3 d7 = pj7.xyz - ri;

            float r2_0 = dot(d0, d0) + eps2;
            float r2_1 = dot(d1, d1) + eps2;
            float r2_2 = dot(d2, d2) + eps2;
            float r2_3 = dot(d3, d3) + eps2;
            float r2_4 = dot(d4, d4) + eps2;
            float r2_5 = dot(d5, d5) + eps2;
            float r2_6 = dot(d6, d6) + eps2;
            float r2_7 = dot(d7, d7) + eps2;

            float inv_r_0 = rsqrt(r2_0);
            float inv_r_1 = rsqrt(r2_1);
            float inv_r_2 = rsqrt(r2_2);
            float inv_r_3 = rsqrt(r2_3);
            float inv_r_4 = rsqrt(r2_4);
            float inv_r_5 = rsqrt(r2_5);
            float inv_r_6 = rsqrt(r2_6);
            float inv_r_7 = rsqrt(r2_7);

            float m_ir_0 = pj0.w * inv_r_0;
            float m_ir_1 = pj1.w * inv_r_1;
            float m_ir_2 = pj2.w * inv_r_2;
            float m_ir_3 = pj3.w * inv_r_3;
            float m_ir_4 = pj4.w * inv_r_4;
            float m_ir_5 = pj5.w * inv_r_5;
            float m_ir_6 = pj6.w * inv_r_6;
            float m_ir_7 = pj7.w * inv_r_7;

            float ir2_0 = inv_r_0 * inv_r_0;
            float ir2_1 = inv_r_1 * inv_r_1;
            float ir2_2 = inv_r_2 * inv_r_2;
            float ir2_3 = inv_r_3 * inv_r_3;
            float ir2_4 = inv_r_4 * inv_r_4;
            float ir2_5 = inv_r_5 * inv_r_5;
            float ir2_6 = inv_r_6 * inv_r_6;
            float ir2_7 = inv_r_7 * inv_r_7;

            float m_inv_r3_0 = m_ir_0 * ir2_0;
            float m_inv_r3_1 = m_ir_1 * ir2_1;
            float m_inv_r3_2 = m_ir_2 * ir2_2;
            float m_inv_r3_3 = m_ir_3 * ir2_3;
            float m_inv_r3_4 = m_ir_4 * ir2_4;
            float m_inv_r3_5 = m_ir_5 * ir2_5;
            float m_inv_r3_6 = m_ir_6 * ir2_6;
            float m_inv_r3_7 = m_ir_7 * ir2_7;

            a += d0 * m_inv_r3_0;
            a += d1 * m_inv_r3_1;
            a += d2 * m_inv_r3_2;
            a += d3 * m_inv_r3_3;
            a += d4 * m_inv_r3_4;
            a += d5 * m_inv_r3_5;
            a += d6 * m_inv_r3_6;
            a += d7 * m_inv_r3_7;
        }
        
        // Handle tail loop
        for (; k < tile_end; ++k) {
            float4 pj = shared_pos[k];
            float3 d  = pj.xyz - ri;
            float r2  = dot(d, d) + eps2;
            
            float inv_r = rsqrt(r2);
            float m_ir = pj.w * inv_r;
            float ir2 = inv_r * inv_r;
            
            a += d * (m_ir * ir2);
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

Result of previous attempt:
  COMPILE FAILED: Error Domain=MTLLibraryErrorDomain Code=3 "program_source:15:66: error: 'max_total_threads_per_threadgroup' attribute cannot be applied to types
                       uint tsize [[threads_per_threadgroup]]) [[max_total_threads_per_threadgroup(1024)]] {
                                                                 ^
" UserInfo={NSLocalizedDescription=program_source:15:66: error: 'max_total_threads_per_threadgroup' attribute cannot be applied to types
                       uint tsize [[threads_per_threadgroup]]) [[max_total_threads_per_threadgroup(1024)]] {
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

Incumbent result:
            256_10: correct, 0.24 ms, 53.9 GFLOPS (1.2% of 4500 GFLOPS)
           1024_10: correct, 1.37 ms, 152.6 GFLOPS (3.4% of 4500 GFLOPS)
           2048_10: correct, 0.86 ms, 980.6 GFLOPS (21.8% of 4500 GFLOPS)
  score (gmean of fraction): 0.0446

## History

- iter  0: compile=OK | correct=True | score=0.02036964781168574
- iter  1: compile=OK | correct=True | score=0.0445744530902113
- iter  2: compile=OK | correct=True | score=0.030858532408641868
- iter  3: compile=FAIL | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
