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

constant uint TILE_SIZE = 256;

[[max_total_threads_per_threadgroup(256)]]
kernel void nbody_step(device const float4 *pos_in  [[buffer(0)]],
                       device       float4 *pos_out [[buffer(1)]],
                       device const float4 *vel_in  [[buffer(2)]],
                       device       float4 *vel_out [[buffer(3)]],
                       device const float  *mass    [[buffer(4)]],
                       constant uint        &N      [[buffer(5)]],
                       constant float       &dt     [[buffer(6)]],
                       constant float       &eps    [[buffer(7)]],
                       constant float       &G      [[buffer(8)]],
                       uint i                       [[thread_position_in_grid]],
                       uint li                      [[thread_position_in_threadgroup]])
{
    threadgroup float4 tg_pos[TILE_SIZE];
    threadgroup float  tg_mass[TILE_SIZE];

    const float3 ri   = (i < N) ? pos_in[i].xyz  : float3(0.0f);
    const float3 vi   = (i < N) ? vel_in[i].xyz  : float3(0.0f);
    float3 a    = float3(0.0f);
    const float  eps2 = eps * eps;

    const uint num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < num_tiles; ++t) {
        uint src = t * TILE_SIZE + li;
        if (src < N) {
            tg_pos[li]  = pos_in[src];
            tg_mass[li] = mass[src];
        } else {
            tg_pos[li]  = float4(0.0f);
            tg_mass[li] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint tile_start = t * TILE_SIZE;
        uint tile_count = min(TILE_SIZE, N - tile_start);

        // Unrolled 8x loop
        uint jj = 0;
        for (; jj + 7 < tile_count; jj += 8) {
            float3 d0 = tg_pos[jj+0].xyz - ri;
            float3 d1 = tg_pos[jj+1].xyz - ri;
            float3 d2 = tg_pos[jj+2].xyz - ri;
            float3 d3 = tg_pos[jj+3].xyz - ri;
            float3 d4 = tg_pos[jj+4].xyz - ri;
            float3 d5 = tg_pos[jj+5].xyz - ri;
            float3 d6 = tg_pos[jj+6].xyz - ri;
            float3 d7 = tg_pos[jj+7].xyz - ri;

            float r0sq = dot(d0, d0) + eps2;
            float r1sq = dot(d1, d1) + eps2;
            float r2sq = dot(d2, d2) + eps2;
            float r3sq = dot(d3, d3) + eps2;
            float r4sq = dot(d4, d4) + eps2;
            float r5sq = dot(d5, d5) + eps2;
            float r6sq = dot(d6, d6) + eps2;
            float r7sq = dot(d7, d7) + eps2;

            // Safe: rsqrt(r^2), then cube it
            float s0 = rsqrt(r0sq); float s0c = s0 * s0 * s0;
            float s1 = rsqrt(r1sq); float s1c = s1 * s1 * s1;
            float s2 = rsqrt(r2sq); float s2c = s2 * s2 * s2;
            float s3 = rsqrt(r3sq); float s3c = s3 * s3 * s3;
            float s4 = rsqrt(r4sq); float s4c = s4 * s4 * s4;
            float s5 = rsqrt(r5sq); float s5c = s5 * s5 * s5;
            float s6 = rsqrt(r6sq); float s6c = s6 * s6 * s6;
            float s7 = rsqrt(r7sq); float s7c = s7 * s7 * s7;

            a += d0 * (tg_mass[jj+0] * s0c);
            a += d1 * (tg_mass[jj+1] * s1c);
            a += d2 * (tg_mass[jj+2] * s2c);
            a += d3 * (tg_mass[jj+3] * s3c);
            a += d4 * (tg_mass[jj+4] * s4c);
            a += d5 * (tg_mass[jj+5] * s5c);
            a += d6 * (tg_mass[jj+6] * s6c);
            a += d7 * (tg_mass[jj+7] * s7c);
        }
        // Remainder
        for (; jj < tile_count; ++jj) {
            float3 d  = tg_pos[jj].xyz - ri;
            float  r2 = dot(d, d) + eps2;
            float  s  = rsqrt(r2);
            float  sc = s * s * s;
            a += d * (tg_mass[jj] * sc);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (i >= N) return;

    // Apply G once at the end
    a *= G;

    float3 v_new = vi + a * dt;
    float3 r_new = ri + v_new * dt;

    pos_out[i] = float4(r_new, 0.0f);
    vel_out[i] = float4(v_new, 0.0f);
}
```

Result of previous attempt:
            256_10: INCORRECT (max_abs_pos=nan, tol=0.0019946529269218444)
  fail_reason: correctness failed at size 256_10: max_abs_pos=nan

## Current best (incumbent)

```metal
// Naive seed kernel for all-pairs N-body gravity, one leapfrog step.
//
// Reads pos_in, vel_in, mass; writes pos_out, vel_out. Each thread updates
// one body i:
//   for j in 0..N: a += G m_j (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)
//   v_new = v_in + a * dt
//   x_new = x_in + v_new * dt
//
// Buffer layout:
//   buffer 0: const float4* pos_in   (N bodies; .xyz = position, .w unused)
//   buffer 1: device float4* pos_out
//   buffer 2: const float4* vel_in   (.xyz = velocity, .w unused)
//   buffer 3: device float4* vel_out
//   buffer 4: const float*  mass     (length N)
//   buffer 5: const uint&   N
//   buffer 6: const float&  dt
//   buffer 7: const float&  eps      (softening)
//   buffer 8: const float&  G

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
                       uint i [[thread_position_in_grid]]) {
    if (i >= N) return;

    float3 ri = pos_in[i].xyz;
    float3 vi = vel_in[i].xyz;
    float3 a  = float3(0.0f);

    float eps2 = eps * eps;
    for (uint j = 0; j < N; ++j) {
        float3 rj = pos_in[j].xyz;
        float  mj = mass[j];
        float3 d  = rj - ri;
        float  r2 = dot(d, d) + eps2;
        float  inv_r3 = rsqrt(r2 * r2 * r2);
        a += G * mj * d * inv_r3;
    }

    float3 v_new = vi + a * dt;
    float3 r_new = ri + v_new * dt;

    pos_out[i] = float4(r_new, 0.0f);
    vel_out[i] = float4(v_new, 0.0f);
}
```

Incumbent result:
            256_10: correct, 0.87 ms, 15.1 GFLOPS (0.3% of 4500 GFLOPS)
           1024_10: correct, 1.10 ms, 189.8 GFLOPS (4.2% of 4500 GFLOPS)
           2048_10: correct, 2.40 ms, 349.7 GFLOPS (7.8% of 4500 GFLOPS)
  score (gmean of fraction): 0.0222

## History

- iter  0: compile=OK | correct=True | score=0.022223142274349746
- iter  1: compile=FAIL | correct=False | score=N/A
- iter  2: compile=OK | correct=False | score=N/A
- iter  3: compile=OK | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
