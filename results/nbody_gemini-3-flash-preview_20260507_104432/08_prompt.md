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
                       uint ti [[thread_index_in_threadgroup]],
                       uint tpg [[threads_per_threadgroup]]) {
    
    // Each thread handles exactly one output body
    if (i >= N) return;

    // Cache the current body's position in registers
    const float3 ri = pos_in[i].xyz;
    const float eps2 = eps * eps;
    const float Gdt = G * dt;
    float3 acc = 0.0f;

    // Use threadgroup memory for cooperative tiling across the whole group
    // 256 is the maximum recommended threadgroup size for general compute
    threadgroup float4 shared_tile[256];

    // Main N-squared interaction loop processed in tiles
    for (uint j_tile = 0; j_tile < N; j_tile += tpg) {
        
        // Cooperative load: each thread loads one body's pos + mass into tile
        const uint j_idx = j_tile + ti;
        if (j_idx < N) {
            float4 p = pos_in[j_idx];
            p.w = mass[j_idx]; // Compact mass into float4 padding
            shared_tile[ti] = p;
        } else {
            shared_tile[ti] = 0.0f; // Zero mass ensures no acceleration contribution
        }

        // Wait for all threads to synchronize tile data
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate acceleration from every body in the current tile
        // Unroll the loop to improve arithmetic intensity and hide latency
        #pragma unroll(16)
        for (uint j = 0; j < tpg; ++j) {
            const float4 pj = shared_tile[j];
            const float3 d = pj.xyz - ri;
            
            // Efficient r^2 calculation using FMA
            const float r2 = fma(d.x, d.x, fma(d.y, d.y, fma(d.z, d.z, eps2)));
            const float inv_r = rsqrt(r2);
            const float inv_r3 = inv_r * inv_r * inv_r;
            
            // acc = sum( m_j * (r_j - r_i) / r^3 )
            acc = fma(d, pj.w * inv_r3, acc);
        }

        // Synchronize before the next tile overwrite
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Leapfrog Integration:
    // 1. v_new = v + (G * acc) * dt
    // 2. r_new = r + v_new * dt
    const float3 vi = vel_in[i].xyz;
    const float3 v_next = fma(acc, Gdt, vi);
    const float3 r_next = fma(v_next, dt, ri);

    // Final writeback to global memory
    pos_out[i] = float4(r_next, 0.0f);
    vel_out[i] = float4(v_next, 0.0f);
}
```

Result of previous attempt:
            256_10: correct, 0.39 ms, 33.9 GFLOPS (0.8% of 4500 GFLOPS)
           1024_10: correct, 1.09 ms, 192.9 GFLOPS (4.3% of 4500 GFLOPS)
           2048_10: correct, 1.25 ms, 671.5 GFLOPS (14.9% of 4500 GFLOPS)
  score (gmean of fraction): 0.0364

## Current best (incumbent)

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

Incumbent result:
            256_10: correct, 0.21 ms, 63.1 GFLOPS (1.4% of 4500 GFLOPS)
           1024_10: correct, 0.72 ms, 290.3 GFLOPS (6.5% of 4500 GFLOPS)
           2048_10: correct, 1.11 ms, 757.9 GFLOPS (16.8% of 4500 GFLOPS)
  score (gmean of fraction): 0.0534

## History

- iter  0: compile=OK | correct=True | score=0.023891517629816377
- iter  1: compile=OK | correct=True | score=0.03697383702976217
- iter  2: compile=OK | correct=True | score=0.04445348898341195
- iter  3: compile=OK | correct=True | score=0.0534141971022608
- iter  4: compile=OK | correct=True | score=0.03520879127394095
- iter  5: compile=OK | correct=True | score=0.03058866661465585
- iter  6: compile=OK | correct=True | score=0.014046995837367945
- iter  7: compile=OK | correct=True | score=0.036402956861922374

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
