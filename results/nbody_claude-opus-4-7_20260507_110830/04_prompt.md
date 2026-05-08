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

#define TILE 256u

[[max_total_threads_per_threadgroup(TILE)]]
kernel void nbody_step(device const float4 *pos_in  [[buffer(0)]],
                       device       float4 *pos_out [[buffer(1)]],
                       device const float4 *vel_in  [[buffer(2)]],
                       device       float4 *vel_out [[buffer(3)]],
                       device const float  *mass    [[buffer(4)]],
                       constant uint        &N      [[buffer(5)]],
                       constant float       &dt     [[buffer(6)]],
                       constant float       &eps    [[buffer(7)]],
                       constant float       &G      [[buffer(8)]],
                       uint i   [[thread_position_in_grid]],
                       uint lid [[thread_position_in_threadgroup]])
{
    threadgroup float4 sdata[TILE];   // .xyz = pos, .w = mass

    bool active = (i < N);
    float3 ri = active ? pos_in[i].xyz : float3(0.0f);
    float3 vi = active ? vel_in[i].xyz : float3(0.0f);
    float3 a  = float3(0.0f);
    float  eps2 = eps * eps;

    uint num_tiles = (N + TILE - 1u) / TILE;

    for (uint t = 0; t < num_tiles; ++t) {
        uint j_base = t * TILE;
        uint j_load = j_base + lid;

        if (j_load < N) {
            float3 pj = pos_in[j_load].xyz;
            float  mj = mass[j_load];
            sdata[lid] = float4(pj, mj);
        } else {
            sdata[lid] = float4(0.0f);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint tile_count = min(TILE, N - j_base);

        if (tile_count == TILE) {
            // Fully populated tile: process 4 bodies at a time to expose ILP.
            for (uint k = 0; k < TILE; k += 4) {
                float4 p0 = sdata[k + 0];
                float4 p1 = sdata[k + 1];
                float4 p2 = sdata[k + 2];
                float4 p3 = sdata[k + 3];

                float3 d0 = p0.xyz - ri;
                float3 d1 = p1.xyz - ri;
                float3 d2 = p2.xyz - ri;
                float3 d3 = p3.xyz - ri;

                float r2_0 = fma(d0.x, d0.x, fma(d0.y, d0.y, fma(d0.z, d0.z, eps2)));
                float r2_1 = fma(d1.x, d1.x, fma(d1.y, d1.y, fma(d1.z, d1.z, eps2)));
                float r2_2 = fma(d2.x, d2.x, fma(d2.y, d2.y, fma(d2.z, d2.z, eps2)));
                float r2_3 = fma(d3.x, d3.x, fma(d3.y, d3.y, fma(d3.z, d3.z, eps2)));

                float s0 = p0.w * rsqrt(r2_0 * r2_0 * r2_0);
                float s1 = p1.w * rsqrt(r2_1 * r2_1 * r2_1);
                float s2 = p2.w * rsqrt(r2_2 * r2_2 * r2_2);
                float s3 = p3.w * rsqrt(r2_3 * r2_3 * r2_3);

                a = fma(d0, s0, a);
                a = fma(d1, s1, a);
                a = fma(d2, s2, a);
                a = fma(d3, s3, a);
            }
        } else {
            for (uint k = 0; k < tile_count; ++k) {
                float4 pj = sdata[k];
                float3 d  = pj.xyz - ri;
                float  r2 = fma(d.x, d.x, fma(d.y, d.y, fma(d.z, d.z, eps2)));
                float  inv_r3 = rsqrt(r2 * r2 * r2);
                float  s = pj.w * inv_r3;
                a = fma(d, s, a);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active) {
        a *= G;
        float3 v_new = vi + a * dt;
        float3 r_new = ri + v_new * dt;
        pos_out[i] = float4(r_new, 0.0f);
        vel_out[i] = float4(v_new, 0.0f);
    }
}
```

Result of previous attempt:
            256_10: INCORRECT (max_abs_pos=nan, tol=0.0019946529269218444)
  fail_reason: correctness failed at size 256_10: max_abs_pos=nan

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

#define TILE 128u

[[max_total_threads_per_threadgroup(TILE)]]
kernel void nbody_step(device const float4 *pos_in  [[buffer(0)]],
                       device       float4 *pos_out [[buffer(1)]],
                       device const float4 *vel_in  [[buffer(2)]],
                       device       float4 *vel_out [[buffer(3)]],
                       device const float  *mass    [[buffer(4)]],
                       constant uint        &N      [[buffer(5)]],
                       constant float       &dt     [[buffer(6)]],
                       constant float       &eps    [[buffer(7)]],
                       constant float       &G      [[buffer(8)]],
                       uint i   [[thread_position_in_grid]],
                       uint lid [[thread_position_in_threadgroup]])
{
    threadgroup float4 sdata[TILE];   // .xyz = pos, .w = mass

    bool active = (i < N);
    float3 ri = active ? pos_in[i].xyz : float3(0.0f);
    float3 vi = active ? vel_in[i].xyz : float3(0.0f);
    float3 a  = float3(0.0f);
    float  eps2 = eps * eps;

    uint num_tiles = (N + TILE - 1u) / TILE;

    for (uint t = 0; t < num_tiles; ++t) {
        uint j_base = t * TILE;
        uint j_load = j_base + lid;

        // Cooperative load: one body per thread into shared memory.
        if (j_load < N) {
            float3 pj = pos_in[j_load].xyz;
            float  mj = mass[j_load];
            sdata[lid] = float4(pj, mj);
        } else {
            sdata[lid] = float4(0.0f);   // mass=0 contributes nothing
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint tile_count = min(TILE, N - j_base);

        if (tile_count == TILE) {
            // Fully populated tile: unroll aggressively.
            #pragma unroll(8)
            for (uint k = 0; k < TILE; ++k) {
                float4 pj = sdata[k];
                float3 d  = pj.xyz - ri;
                float  r2 = fma(d.x, d.x, fma(d.y, d.y, fma(d.z, d.z, eps2)));
                float  inv_r3 = rsqrt(r2 * r2 * r2);
                float  s = pj.w * inv_r3;
                a = fma(d, s, a);
            }
        } else {
            for (uint k = 0; k < tile_count; ++k) {
                float4 pj = sdata[k];
                float3 d  = pj.xyz - ri;
                float  r2 = fma(d.x, d.x, fma(d.y, d.y, fma(d.z, d.z, eps2)));
                float  inv_r3 = rsqrt(r2 * r2 * r2);
                float  s = pj.w * inv_r3;
                a = fma(d, s, a);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active) {
        a *= G;
        float3 v_new = vi + a * dt;
        float3 r_new = ri + v_new * dt;
        pos_out[i] = float4(r_new, 0.0f);
        vel_out[i] = float4(v_new, 0.0f);
    }
}
```

Incumbent result:
            256_10: correct, 0.18 ms, 72.3 GFLOPS (1.6% of 4500 GFLOPS)
           1024_10: correct, 0.69 ms, 303.8 GFLOPS (6.8% of 4500 GFLOPS)
           2048_10: correct, 1.21 ms, 691.3 GFLOPS (15.4% of 4500 GFLOPS)
  score (gmean of fraction): 0.0550

## History

- iter  0: compile=OK | correct=True | score=0.019439948447710863
- iter  1: compile=FAIL | correct=False | score=N/A
- iter  2: compile=OK | correct=True | score=0.05503116661875314
- iter  3: compile=OK | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
