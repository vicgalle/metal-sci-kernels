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
                       uint tid [[thread_index_in_threadgroup]],
                       uint tgs [[threadgroup_size_in_threadgroup]]) {
    // Guard against out-of-bounds threads
    if (i >= N) return;

    // Load current particle's state
    const float3 ri = pos_in[i].xyz;
    float3 acc = float3(0.0f);
    const float eps2 = eps * eps;
    
    // Tiling parameters
    const uint TILE_SIZE = 256;
    threadgroup float4 tile[TILE_SIZE];

    for (uint j_base = 0; j_base < N; j_base += TILE_SIZE) {
        // Cooperative load: all threads help fill the tile in threadgroup memory
        for (uint l = tid; l < TILE_SIZE; l += tgs) {
            uint j_idx = j_base + l;
            if (j_idx < N) {
                // Combine position and mass into one float4 for shared storage
                tile[l] = float4(pos_in[j_idx].xyz, mass[j_idx]);
            } else {
                // Pad with zero mass to avoid branching in the inner loop
                tile[l] = float4(0.0f);
            }
        }
        
        // Synchronize threads before computing interactions
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute interactions for this tile (compiler is hinted to unroll)
        #pragma unroll(16)
        for (uint j = 0; j < TILE_SIZE; ++j) {
            float4 rj_mj = tile[j];
            float3 d = rj_mj.xyz - ri;
            float r2 = dot(d, d) + eps2;
            float inv_r = rsqrt(r2);
            float inv_r3 = inv_r * inv_r * inv_r;
            
            // Interaction: a_i += m_j * (r_j - r_i) / (r^2 + eps^2)^(3/2)
            acc += (rj_mj.w * inv_r3) * d;
        }

        // Synchronize before loading the next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Integrate: v_new = v + a*dt; r_new = r + v_new*dt
    const float3 vi = vel_in[i].xyz;
    const float3 a = acc * G;
    const float3 v_new = vi + a * dt;
    const float3 r_new = ri + v_new * dt;

    // Write updated state (w is used for padding)
    pos_out[i] = float4(r_new, 0.0f);
    vel_out[i] = float4(v_new, 0.0f);
}
```

Result of previous attempt:
  COMPILE FAILED: Error Domain=MTLLibraryErrorDomain Code=3 "program_source:15:35: warning: unknown attribute 'threadgroup_size_in_threadgroup' ignored [-Wunknown-attributes]
                       uint tgs [[threadgroup_size_in_threadgroup]]) {
                                  ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
program_source:15:24: error: invalid type 'uint' (aka 'unsigned int') for input declaration in a kernel function
                       uint tgs [[threadgroup_size_in_threadgroup]]) {
                       ^~~~~~~~
" UserInfo={NSLocalizedDescription=program_source:15:35: warning: unknown attribute 'threadgroup_size_in_threadgroup' ignored [-Wunknown-attributes]
                       uint tgs [[threadgroup_size_in_threadgroup]]) {
                                  ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
program_source:15:24: error: invalid type 'uint' (aka 'unsigned int') for input declaration in a kernel function
                       uint tgs [[threadgroup_size_in_threadgroup]]) {
                       ^~~~~~~~
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
                       uint lane_id [[thread_index_in_simdgroup]]) {
    // Guard against out-of-bounds threads
    if (i >= N) return;

    // Load current body's state into registers
    float3 ri = pos_in[i].xyz;
    float3 vi = vel_in[i].xyz;
    float3 acc = float3(0.0f);
    const float eps2 = eps * eps;

    // Iterate over all bodies in tiles of SIMD_WIDTH (32 on Apple Silicon)
    // This optimization reduces global memory reads by 32x.
    for (uint j_base = 0; j_base < N; j_base += 32) {
        // Each thread in the SIMD-group loads one body from global memory
        uint j = j_base + lane_id;
        float4 rj_mj = (j < N) ? float4(pos_in[j].xyz, mass[j]) : float4(0.0f);

        // Share the loaded body with all other threads in the SIMD-group
        for (ushort lane = 0; lane < 32; ++lane) {
            // Only process valid bodies if N is not a multiple of 32
            if (j_base + lane < N) {
                float4 other = simd_broadcast(rj_mj, lane);
                
                float3 d = other.xyz - ri;
                float r2 = d.x * d.x + d.y * d.y + d.z * d.z + eps2;
                
                // Calculate (r^2 + eps^2)^(-1.5) efficiently
                float inv_r = rsqrt(r2);
                float inv_r3 = inv_r * inv_r * inv_r;
                
                // Accumulate weighted acceleration vector
                acc += (other.w * inv_r3) * d;
            }
        }
    }

    // Apply gravity constant and perform Leapfrog integration
    float3 a = acc * G;
    float3 v_new = vi + a * dt;
    float3 r_new = ri + v_new * dt;

    // Write out updated state (w component is preserved as padding)
    pos_out[i] = float4(r_new, 0.0f);
    vel_out[i] = float4(v_new, 0.0f);
}
```

Incumbent result:
            256_10: correct, 0.79 ms, 16.7 GFLOPS (0.4% of 4500 GFLOPS)
           1024_10: correct, 1.38 ms, 152.1 GFLOPS (3.4% of 4500 GFLOPS)
           2048_10: correct, 2.07 ms, 405.7 GFLOPS (9.0% of 4500 GFLOPS)
  score (gmean of fraction): 0.0224

## History

- iter  0: compile=OK | correct=True | score=0.020228761441132028
- iter  1: compile=OK | correct=True | score=0.022442939800843845
- iter  2: compile=FAIL | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
