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
    
    // Shared memory for cooperative loading of bodies (position.xyz, mass.w)
    threadgroup float4 shared_tile[256];

    float3 ri = 0.0f;
    if (i < N) {
        ri = pos_in[i].xyz;
    }

    float3 acc = 0.0f;
    const float eps2 = eps * eps;

    // Loop over all bodies in tiles of size tpg
    for (uint j_tile = 0; j_tile < N; j_tile += tpg) {
        // Cooperative load: each thread in the group loads one body's data
        const uint j_idx = j_tile + ti;
        if (j_idx < N) {
            float4 p = pos_in[j_idx];
            p.w = mass[j_idx];
            shared_tile[ti] = p;
        } else {
            shared_tile[ti] = 0.0f; // Mass 0 ensures no force contribution
        }

        // Ensure all threads have finished loading the current tile
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate gravitational interaction from all bodies in this tile
        if (i < N) {
            #pragma unroll(8)
            for (uint j = 0; j < tpg; ++j) {
                float4 pj = shared_tile[j];
                float3 d = pj.xyz - ri;
                float r2 = d.x * d.x + d.y * d.y + d.z * d.z + eps2;
                float inv_r3 = rsqrt(r2 * r2 * r2);
                acc += (pj.w * inv_r3) * d;
            }
        }

        // Synchronize before loading the next tile to avoid overwriting data
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Update position and velocity using leapfrog integration
    if (i < N) {
        const float3 vi = vel_in[i].xyz;
        const float3 v_new = vi + (G * dt) * acc;
        const float3 r_new = ri + v_new * dt;

        pos_out[i] = float4(r_new, 0.0f);
        vel_out[i] = float4(v_new, 0.0f);
    }
}
```

Result of previous attempt:
            256_10: correct, 0.39 ms, 33.5 GFLOPS (0.7% of 4500 GFLOPS)
           1024_10: correct, 1.10 ms, 189.9 GFLOPS (4.2% of 4500 GFLOPS)
           2048_10: correct, 1.16 ms, 724.6 GFLOPS (16.1% of 4500 GFLOPS)
  score (gmean of fraction): 0.0370

## History

- iter  0: compile=OK | correct=True | score=0.023891517629816377
- iter  1: compile=OK | correct=True | score=0.03697383702976217

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
