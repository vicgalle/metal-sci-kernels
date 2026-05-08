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

#define NBODY_TILE_SIZE 512

static inline float3 nbody_accum_pair(float3 acc, float3 ri, float4 pm, float eps2) {
    float3 d = pm.xyz - ri;
    float r2 = dot(d, d) + eps2;
    float inv_r = rsqrt(r2);
    float inv_r3 = inv_r * inv_r * inv_r;
    return acc + d * (pm.w * inv_r3);
}

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
                       uint3 tptg [[threads_per_threadgroup]])
                       [[max_total_threads_per_threadgroup(64)]] {
    threadgroup float4 tile[NBODY_TILE_SIZE];

    const float eps2 = eps * eps;
    const float gdt = G * dt;

    // For very small N, avoid barriers/threadgroup traffic; just improve ILP.
    if (N <= (uint)NBODY_TILE_SIZE) {
        if (i >= N) return;

        float3 ri = pos_in[i].xyz;
        float3 vi = vel_in[i].xyz;

        float3 acc0 = float3(0.0f);
        float3 acc1 = float3(0.0f);
        float3 acc2 = float3(0.0f);
        float3 acc3 = float3(0.0f);

        uint j = 0;
        for (; j + 3u < N; j += 4u) {
            float4 pm0 = pos_in[j + 0u]; pm0.w = mass[j + 0u];
            float4 pm1 = pos_in[j + 1u]; pm1.w = mass[j + 1u];
            float4 pm2 = pos_in[j + 2u]; pm2.w = mass[j + 2u];
            float4 pm3 = pos_in[j + 3u]; pm3.w = mass[j + 3u];

            acc0 = nbody_accum_pair(acc0, ri, pm0, eps2);
            acc1 = nbody_accum_pair(acc1, ri, pm1, eps2);
            acc2 = nbody_accum_pair(acc2, ri, pm2, eps2);
            acc3 = nbody_accum_pair(acc3, ri, pm3, eps2);
        }

        for (; j < N; ++j) {
            float4 pm = pos_in[j];
            pm.w = mass[j];
            acc0 = nbody_accum_pair(acc0, ri, pm, eps2);
        }

        float3 acc = (acc0 + acc1) + (acc2 + acc3);
        float3 v_new = vi + acc * gdt;
        float3 r_new = ri + v_new * dt;

        pos_out[i] = float4(r_new, 0.0f);
        vel_out[i] = float4(v_new, 0.0f);
        return;
    }

    const uint tg_size = tptg.x * tptg.y * tptg.z;
    const uint group_start = i - tid;

    // Uniform whole-threadgroup exit; safe before barriers.
    if (group_start >= N) return;

    const bool active = (i < N);
    float3 ri = active ? pos_in[i].xyz : float3(0.0f);
    float3 vi = active ? vel_in[i].xyz : float3(0.0f);

    float3 acc0 = float3(0.0f);
    float3 acc1 = float3(0.0f);
    float3 acc2 = float3(0.0f);
    float3 acc3 = float3(0.0f);

    for (uint base = 0; base < N; base += (uint)NBODY_TILE_SIZE) {
        // Cooperative load: all threads fill a 512-body source tile.
        for (uint l = tid; l < (uint)NBODY_TILE_SIZE; l += tg_size) {
            uint src = base + l;
            if (src < N) {
                float4 pm = pos_in[src];
                pm.w = mass[src];
                tile[l] = pm;
            } else {
                tile[l] = float4(0.0f);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            uint remaining = N - base;
            uint count = remaining < (uint)NBODY_TILE_SIZE ? remaining : (uint)NBODY_TILE_SIZE;

            if (count == (uint)NBODY_TILE_SIZE) {
                for (uint k = 0; k < (uint)NBODY_TILE_SIZE; k += 4u) {
                    float4 pm0 = tile[k + 0u];
                    float4 pm1 = tile[k + 1u];
                    float4 pm2 = tile[k + 2u];
                    float4 pm3 = tile[k + 3u];

                    acc0 = nbody_accum_pair(acc0, ri, pm0, eps2);
                    acc1 = nbody_accum_pair(acc1, ri, pm1, eps2);
                    acc2 = nbody_accum_pair(acc2, ri, pm2, eps2);
                    acc3 = nbody_accum_pair(acc3, ri, pm3, eps2);
                }
            } else {
                uint k = 0;
                for (; k + 3u < count; k += 4u) {
                    float4 pm0 = tile[k + 0u];
                    float4 pm1 = tile[k + 1u];
                    float4 pm2 = tile[k + 2u];
                    float4 pm3 = tile[k + 3u];

                    acc0 = nbody_accum_pair(acc0, ri, pm0, eps2);
                    acc1 = nbody_accum_pair(acc1, ri, pm1, eps2);
                    acc2 = nbody_accum_pair(acc2, ri, pm2, eps2);
                    acc3 = nbody_accum_pair(acc3, ri, pm3, eps2);
                }

                for (; k < count; ++k) {
                    acc0 = nbody_accum_pair(acc0, ri, tile[k], eps2);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active) {
        float3 acc = (acc0 + acc1) + (acc2 + acc3);
        float3 v_new = vi + acc * gdt;
        float3 r_new = ri + v_new * dt;

        pos_out[i] = float4(r_new, 0.0f);
        vel_out[i] = float4(v_new, 0.0f);
    }
}
```

Result of previous attempt:
  COMPILE FAILED: Error Domain=MTLLibraryErrorDomain Code=3 "program_source:26:26: error: 'max_total_threads_per_threadgroup' attribute cannot be applied to types
                       [[max_total_threads_per_threadgroup(64)]] {
                         ^
program_source:14:13: error: expecting input declarations with either all scalar types or all vector types with the same number of elements
kernel void nbody_step(device const float4 *pos_in  [[buffer(0)]],
            ^
program_source:23:29: note: declaration with attribute 'thread_position_in_grid' of type 'uint' (aka 'unsigned int') here
                       uint i [[thread_position_in_grid]],
                       ~~~~~^
program_source:25:30: note: declaration with attribute 'threads_per_threadgroup' of type 'uint3' (vector of 3 'unsigned int' values) here
                       uint3 tptg [[threads_per_threadgroup]])
                       ~~~~~~^~~~
" UserInfo={NSLocalizedDescription=program_source:26:26: error: 'max_total_threads_per_threadgroup' attribute cannot be applied to types
                       [[max_total_threads_per_threadgroup(64)]] {
                         ^
program_source:14:13: error: expecting input declarations with either all scalar types or all vector types with the same number of elements
kernel void nbody_step(device const float4 *pos_in  [[buffer(0)]],
            ^
program_source:23:29: note: declaration with attribute 'thread_position_in_grid' of type 'uint' (aka 'unsigned int') here
                       uint i [[thread_position_in_grid]],
                       ~~~~~^
program_source:25:30: note: declaration with attribute 'threads_per_threadgroup' of type 'uint3' (vector of 3 'unsigned int' values) here
                       uint3 tptg [[threads_per_threadgroup]])
                       ~~~~~~^~~~
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
           1024_10: correct, 1.45 ms, 145.1 GFLOPS (3.2% of 4500 GFLOPS)
           2048_10: correct, 2.40 ms, 349.4 GFLOPS (7.8% of 4500 GFLOPS)
  score (gmean of fraction): 0.0203

## History

- iter  0: compile=OK | correct=True | score=0.02032597394032279
- iter  1: compile=FAIL | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
