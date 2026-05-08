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

#define TILE 128

kernel void nbody_step(device const float4 *pos_in  [[buffer(0)]],
                       device       float4 *pos_out [[buffer(1)]],
                       device const float4 *vel_in  [[buffer(2)]],
                       device       float4 *vel_out [[buffer(3)]],
                       device const float  *mass    [[buffer(4)]],
                       constant uint        &N      [[buffer(5)]],
                       constant float       &dt     [[buffer(6)]],
                       constant float       &eps    [[buffer(7)]],
                       constant float       &G      [[buffer(8)]],
                       uint  i   [[thread_position_in_grid]],
                       uint  lid [[thread_position_in_threadgroup]],
                       uint  tgs [[threads_per_threadgroup]]) [[max_total_threads_per_threadgroup(TILE)]]
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

        uint tile_count = min((uint)TILE, N - j_base);

        #pragma unroll(8)
        for (uint k = 0; k < tile_count; ++k) {
            float4 pj = sdata[k];
            float3 d  = pj.xyz - ri;
            float  r2 = fma(d.x, d.x, fma(d.y, d.y, fma(d.z, d.z, eps2)));
            float  inv_r3 = rsqrt(r2 * r2 * r2);
            float  s = pj.w * inv_r3;
            a = fma(d, s, a);
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
  COMPILE FAILED: Error Domain=MTLLibraryErrorDomain Code=3 "program_source:17:65: error: 'max_total_threads_per_threadgroup' attribute cannot be applied to types
                       uint  tgs [[threads_per_threadgroup]]) [[max_total_threads_per_threadgroup(TILE)]]
                                                                ^
" UserInfo={NSLocalizedDescription=program_source:17:65: error: 'max_total_threads_per_threadgroup' attribute cannot be applied to types
                       uint  tgs [[threads_per_threadgroup]]) [[max_total_threads_per_threadgroup(TILE)]]
                                                                ^
}

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
           1024_10: correct, 1.67 ms, 125.3 GFLOPS (2.8% of 4500 GFLOPS)
           2048_10: correct, 2.38 ms, 353.0 GFLOPS (7.8% of 4500 GFLOPS)
  score (gmean of fraction): 0.0194

## History

- iter  0: compile=OK | correct=True | score=0.019439948447710863
- iter  1: compile=FAIL | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
