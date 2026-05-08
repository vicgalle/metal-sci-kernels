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
    
    // Allocate threadgroup memory for cooperative loading of positions and masses
    // A tile size of 256 is efficient for Apple Silicon SIMD width (32)
    threadgroup float4 shared_tile[256];

    float3 ri = 0.0f;
    if (i < N) {
        ri = pos_in[i].xyz;
    }

    float3 acc = 0.0f;
    const float eps2 = eps * eps;

    // Iterate through all bodies in blocks of tpg
    for (uint j_tile = 0; j_tile < N; j_tile += tpg) {
        // Cooperative load: all threads in the group load one body into shared memory
        const uint j_idx = j_tile + ti;
        if (j_idx < N) {
            float4 p = pos_in[j_idx];
            p.w = mass[j_idx];
            shared_tile[ti] = p;
        } else {
            shared_tile[ti] = 0.0f; // Mass 0 ensures no gravitational contribution
        }

        // Synchronize threads to ensure the entire tile is loaded
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate acceleration from the current tile
        if (i < N) {
            #pragma unroll(8)
            for (uint j = 0; j < tpg; ++j) {
                float4 pj = shared_tile[j];
                float3 d = pj.xyz - ri;
                float r2 = d.x * d.x + d.y * d.y + d.z * d.z + eps2;
                float inv_r = rsqrt(r2);
                float inv_r3 = inv_r * inv_r * inv_r;
                acc += (pj.w * inv_r3) * d;
            }
        }

        // Synchronize before loading the next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Update state using Leapfrog integration: v = v + a*dt, r = r + v*dt
    if (i < N) {
        const float3 vi = vel_in[i].xyz;
        const float3 v_next = vi + (G * dt) * acc;
        const float3 r_next = ri + v_next * dt;

        pos_out[i] = float4(r_next, 0.0f);
        vel_out[i] = float4(v_next, 0.0f);
    }
}
```

Result of previous attempt:
            256_10: correct, 0.20 ms, 65.9 GFLOPS (1.5% of 4500 GFLOPS)
           1024_10: correct, 1.35 ms, 155.1 GFLOPS (3.4% of 4500 GFLOPS)
           2048_10: correct, 1.07 ms, 783.2 GFLOPS (17.4% of 4500 GFLOPS)
  score (gmean of fraction): 0.0445

## History

- iter  0: compile=OK | correct=True | score=0.023891517629816377
- iter  1: compile=OK | correct=True | score=0.03697383702976217
- iter  2: compile=OK | correct=True | score=0.04445348898341195

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
