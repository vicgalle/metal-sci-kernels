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
    
    // Boundary check
    if (i >= N) return;

    // Cache current body state into registers
    const float3 ri = pos_in[i].xyz;
    const float3 vi = vel_in[i].xyz;
    float3 acc = float3(0.0f);
    const float eps2 = eps * eps;
    
    // Shared memory for cooperative tiling
    threadgroup float4 tile[256];

    // Iterate over all bodies in the system using tiles
    for (uint j_tile = 0; j_tile < N; j_tile += tpg) {
        // Cooperative load: each thread fetches one body into shared memory
        const uint j_idx = j_tile + ti;
        if (j_idx < N) {
            float4 p = pos_in[j_idx];
            // Pre-multiply mass by G here to save N*N multiplications in the loop
            p.w = mass[j_idx] * G;
            tile[ti] = p;
        } else {
            tile[ti] = float4(0.0f);
        }

        // Synchronize threads to ensure the tile is fully loaded
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate acceleration from all bodies in the current tile
        // Unroll inner loop to hide latency of rsqrt and FMA
        #pragma unroll(8)
        for (uint j = 0; j < tpg; ++j) {
            const float4 pj = tile[j];
            const float3 d = pj.xyz - ri;
            
            // Standard N-body gravity calculation with softening
            const float r2 = dot(d, d) + eps2;
            const float inv_r = rsqrt(r2);
            const float inv_r3 = (inv_r * inv_r) * inv_r;
            
            // acc = sum( G * m_j * (r_j - r_i) / |r_j - r_i|^3 )
            acc = fma(d, pj.w * inv_r3, acc);
        }

        // Synchronize before loading the next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Leapfrog Integration: 
    // v_new = v + a * dt
    // r_new = r + v_new * dt
    const float3 v_next = fma(acc, dt, vi);
    const float3 r_next = fma(v_next, dt, ri);

    // Write back results to global memory
    pos_out[i] = float4(r_next, 0.0f);
    vel_out[i] = float4(v_next, 0.0f);
}
```

Result of previous attempt:
            256_10: correct, 0.39 ms, 33.7 GFLOPS (0.7% of 4500 GFLOPS)
           1024_10: correct, 1.36 ms, 154.0 GFLOPS (3.4% of 4500 GFLOPS)
           2048_10: correct, 1.67 ms, 502.9 GFLOPS (11.2% of 4500 GFLOPS)
  score (gmean of fraction): 0.0306

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

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
