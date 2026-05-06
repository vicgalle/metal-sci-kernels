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
                       uint lane_id [[thread_index_in_simdgroup]]) {
    // Guard against out-of-bounds threads in the 1D grid
    if (i >= N) return;

    // Load current particle state into registers
    const float3 ri = pos_in[i].xyz;
    const float3 vi = vel_in[i].xyz;
    const float eps2 = eps * eps;
    float3 acc = float3(0.0f);

    uint j_base = 0;
    // Main Loop: Process bodies in blocks of 32 (SIMD width)
    // This part is branch-free for maximum throughput.
    for (; j_base + 32 <= N; j_base += 32) {
        // Cooperative load: each thread in the SIMD-group loads one body
        uint j_load = j_base + lane_id;
        float4 rj_mj = float4(pos_in[j_load].xyz, mass[j_load]);
        
        #pragma unroll(32)
        for (ushort l = 0; l < 32; ++l) {
            // Broadcast the loaded body's position and mass to all lanes
            float4 other = simd_broadcast(rj_mj, l);
            float3 d = other.xyz - ri;
            float r2 = dot(d, d) + eps2;
            
            // Efficiently compute (r^2 + eps^2)^(-1.5)
            float inv_r = rsqrt(r2);
            float inv_r3 = inv_r * inv_r * inv_r;
            
            // Accumulate acceleration (mass is stored in .w)
            acc += d * (other.w * inv_r3);
        }
    }

    // Tail Loop: Handle the remaining particles if N is not a multiple of 32
    if (j_base < N) {
        uint j_load = j_base + lane_id;
        float4 rj_mj = (j_load < N) ? float4(pos_in[j_load].xyz, mass[j_load]) : float4(0.0f);
        
        #pragma unroll(32)
        for (ushort l = 0; l < 32; ++l) {
            if (j_base + l < N) {
                float4 other = simd_broadcast(rj_mj, l);
                float3 d = other.xyz - ri;
                float r2 = dot(d, d) + eps2;
                float inv_r = rsqrt(r2);
                float inv_r3 = inv_r * inv_r * inv_r;
                acc += d * (other.w * inv_r3);
            }
        }
    }

    // Leapfrog Integration:
    // a = acc * G
    // v_new = v + a * dt
    // r_new = r + v_new * dt
    const float3 a = acc * G;
    const float3 v_new = vi + a * dt;
    const float3 r_new = ri + v_new * dt;

    // Write-back updated state; .w component is preserved as padding
    pos_out[i] = float4(r_new, 0.0f);
    vel_out[i] = float4(v_new, 0.0f);
}
```

Result of previous attempt:
            256_10: correct, 0.38 ms, 34.8 GFLOPS (0.8% of 4500 GFLOPS)
           1024_10: correct, 0.69 ms, 304.3 GFLOPS (6.8% of 4500 GFLOPS)
           2048_10: correct, 1.26 ms, 666.6 GFLOPS (14.8% of 4500 GFLOPS)
  score (gmean of fraction): 0.0426

## History

- iter  0: compile=OK | correct=True | score=0.020228761441132028
- iter  1: compile=OK | correct=True | score=0.022442939800843845
- iter  2: compile=FAIL | correct=False | score=N/A
- iter  3: compile=FAIL | correct=False | score=N/A
- iter  4: compile=OK | correct=True | score=0.042641989903706906

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
