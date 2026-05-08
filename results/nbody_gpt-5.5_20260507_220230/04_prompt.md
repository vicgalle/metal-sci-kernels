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

static inline float3 nbody_accum_one(float3 acc, float3 ri, float4 pm, float eps2) {
    const float dx = pm.x - ri.x;
    const float dy = pm.y - ri.y;
    const float dz = pm.z - ri.z;

    const float r2 = fma(dx, dx, fma(dy, dy, fma(dz, dz, eps2)));
    const float inv_r = rsqrt(r2);
    const float inv_r2 = inv_r * inv_r;
    const float s = pm.w * inv_r * inv_r2;

    acc.x = fma(dx, s, acc.x);
    acc.y = fma(dy, s, acc.y);
    acc.z = fma(dz, s, acc.z);
    return acc;
}

static inline float4 nbody_shuffle4(float4 v, ushort lane) {
    return float4(simd_shuffle(v.x, lane),
                  simd_shuffle(v.y, lane),
                  simd_shuffle(v.z, lane),
                  simd_shuffle(v.w, lane));
}

#define NBODY_ACCUM_SHFL(ACC, LANE_CONST)                 \
{                                                         \
    float4 q = nbody_shuffle4(pm, (ushort)(LANE_CONST));  \
    (ACC) = nbody_accum_one((ACC), ri, q, eps2);          \
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
                       ushort lane [[thread_index_in_simdgroup]],
                       uint tg_size [[threads_per_threadgroup]])
                       [[max_total_threads_per_threadgroup(32)]]
{
    const float eps2 = eps * eps;
    const float gdt = G * dt;

    if (((N & 31u) != 0u) || ((tg_size & 31u) != 0u)) {
        if (i >= N) return;

        const float3 ri = pos_in[i].xyz;
        const float3 vi = vel_in[i].xyz;

        float3 acc0 = float3(0.0f);
        float3 acc1 = float3(0.0f);
        float3 acc2 = float3(0.0f);
        float3 acc3 = float3(0.0f);

        uint j = 0;
        for (; j + 7u < N; j += 8u) {
            float4 pm0 = pos_in[j + 0u]; pm0.w = mass[j + 0u];
            float4 pm1 = pos_in[j + 1u]; pm1.w = mass[j + 1u];
            float4 pm2 = pos_in[j + 2u]; pm2.w = mass[j + 2u];
            float4 pm3 = pos_in[j + 3u]; pm3.w = mass[j + 3u];
            float4 pm4 = pos_in[j + 4u]; pm4.w = mass[j + 4u];
            float4 pm5 = pos_in[j + 5u]; pm5.w = mass[j + 5u];
            float4 pm6 = pos_in[j + 6u]; pm6.w = mass[j + 6u];
            float4 pm7 = pos_in[j + 7u]; pm7.w = mass[j + 7u];

            acc0 = nbody_accum_one(acc0, ri, pm0, eps2);
            acc1 = nbody_accum_one(acc1, ri, pm1, eps2);
            acc2 = nbody_accum_one(acc2, ri, pm2, eps2);
            acc3 = nbody_accum_one(acc3, ri, pm3, eps2);
            acc0 = nbody_accum_one(acc0, ri, pm4, eps2);
            acc1 = nbody_accum_one(acc1, ri, pm5, eps2);
            acc2 = nbody_accum_one(acc2, ri, pm6, eps2);
            acc3 = nbody_accum_one(acc3, ri, pm7, eps2);
        }

        for (; j + 3u < N; j += 4u) {
            float4 pm0 = pos_in[j + 0u]; pm0.w = mass[j + 0u];
            float4 pm1 = pos_in[j + 1u]; pm1.w = mass[j + 1u];
            float4 pm2 = pos_in[j + 2u]; pm2.w = mass[j + 2u];
            float4 pm3 = pos_in[j + 3u]; pm3.w = mass[j + 3u];

            acc0 = nbody_accum_one(acc0, ri, pm0, eps2);
            acc1 = nbody_accum_one(acc1, ri, pm1, eps2);
            acc2 = nbody_accum_one(acc2, ri, pm2, eps2);
            acc3 = nbody_accum_one(acc3, ri, pm3, eps2);
        }

        for (; j < N; ++j) {
            float4 pm = pos_in[j];
            pm.w = mass[j];
            acc0 = nbody_accum_one(acc0, ri, pm, eps2);
        }

        const float3 acc = (acc0 + acc1) + (acc2 + acc3);
        const float3 v_new = vi + acc * gdt;
        const float3 r_new = ri + v_new * dt;

        pos_out[i] = float4(r_new, 0.0f);
        vel_out[i] = float4(v_new, 0.0f);
        return;
    }

    const uint simd_start = i - (uint)lane;
    if (simd_start >= N) return;

    const float3 ri = pos_in[i].xyz;
    const float3 vi = vel_in[i].xyz;

    float3 acc0 = float3(0.0f);
    float3 acc1 = float3(0.0f);
    float3 acc2 = float3(0.0f);
    float3 acc3 = float3(0.0f);
    float3 acc4 = float3(0.0f);
    float3 acc5 = float3(0.0f);
    float3 acc6 = float3(0.0f);
    float3 acc7 = float3(0.0f);

    for (uint base = 0; base < N; base += 32u) {
        const uint src = base + (uint)lane;
        float4 pm = pos_in[src];
        pm.w = mass[src];

        NBODY_ACCUM_SHFL(acc0,  0)
        NBODY_ACCUM_SHFL(acc1,  1)
        NBODY_ACCUM_SHFL(acc2,  2)
        NBODY_ACCUM_SHFL(acc3,  3)
        NBODY_ACCUM_SHFL(acc4,  4)
        NBODY_ACCUM_SHFL(acc5,  5)
        NBODY_ACCUM_SHFL(acc6,  6)
        NBODY_ACCUM_SHFL(acc7,  7)

        NBODY_ACCUM_SHFL(acc0,  8)
        NBODY_ACCUM_SHFL(acc1,  9)
        NBODY_ACCUM_SHFL(acc2, 10)
        NBODY_ACCUM_SHFL(acc3, 11)
        NBODY_ACCUM_SHFL(acc4, 12)
        NBODY_ACCUM_SHFL(acc5, 13)
        NBODY_ACCUM_SHFL(acc6, 14)
        NBODY_ACCUM_SHFL(acc7, 15)

        NBODY_ACCUM_SHFL(acc0, 16)
        NBODY_ACCUM_SHFL(acc1, 17)
        NBODY_ACCUM_SHFL(acc2, 18)
        NBODY_ACCUM_SHFL(acc3, 19)
        NBODY_ACCUM_SHFL(acc4, 20)
        NBODY_ACCUM_SHFL(acc5, 21)
        NBODY_ACCUM_SHFL(acc6, 22)
        NBODY_ACCUM_SHFL(acc7, 23)

        NBODY_ACCUM_SHFL(acc0, 24)
        NBODY_ACCUM_SHFL(acc1, 25)
        NBODY_ACCUM_SHFL(acc2, 26)
        NBODY_ACCUM_SHFL(acc3, 27)
        NBODY_ACCUM_SHFL(acc4, 28)
        NBODY_ACCUM_SHFL(acc5, 29)
        NBODY_ACCUM_SHFL(acc6, 30)
        NBODY_ACCUM_SHFL(acc7, 31)
    }

    const float3 acc = ((acc0 + acc1) + (acc2 + acc3)) +
                       ((acc4 + acc5) + (acc6 + acc7));

    const float3 v_new = vi + acc * gdt;
    const float3 r_new = ri + v_new * dt;

    pos_out[i] = float4(r_new, 0.0f);
    vel_out[i] = float4(v_new, 0.0f);
}

#undef NBODY_ACCUM_SHFL
```

Result of previous attempt:
  COMPILE FAILED: Error Domain=MTLLibraryErrorDomain Code=3 "program_source:45:26: error: 'max_total_threads_per_threadgroup' attribute cannot be applied to types
                       [[max_total_threads_per_threadgroup(32)]]
                         ^
" UserInfo={NSLocalizedDescription=program_source:45:26: error: 'max_total_threads_per_threadgroup' attribute cannot be applied to types
                       [[max_total_threads_per_threadgroup(32)]]
                         ^
}

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

#define NBODY_TILE_SIZE 512u

static inline float3 nbody_accum_one(float3 acc, float3 ri, float4 pm, float eps2) {
    float3 d = pm.xyz - ri;
    float r2 = dot(d, d) + eps2;
    float inv_r = rsqrt(r2);
    float inv_r2 = inv_r * inv_r;
    float s = pm.w * inv_r * inv_r2;
    return acc + d * s;
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
                       uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float4 tile[NBODY_TILE_SIZE];

    const float eps2 = eps * eps;
    const float gdt = G * dt;

    // Small problem path: avoid threadgroup barriers and just improve ILP.
    if (N <= NBODY_TILE_SIZE) {
        if (i >= N) return;

        float3 ri = pos_in[i].xyz;
        float3 vi = vel_in[i].xyz;

        float3 acc0 = float3(0.0f);
        float3 acc1 = float3(0.0f);
        float3 acc2 = float3(0.0f);
        float3 acc3 = float3(0.0f);

        uint j = 0;
        for (; j + 7u < N; j += 8u) {
            float4 pm0 = pos_in[j + 0u]; pm0.w = mass[j + 0u];
            float4 pm1 = pos_in[j + 1u]; pm1.w = mass[j + 1u];
            float4 pm2 = pos_in[j + 2u]; pm2.w = mass[j + 2u];
            float4 pm3 = pos_in[j + 3u]; pm3.w = mass[j + 3u];
            float4 pm4 = pos_in[j + 4u]; pm4.w = mass[j + 4u];
            float4 pm5 = pos_in[j + 5u]; pm5.w = mass[j + 5u];
            float4 pm6 = pos_in[j + 6u]; pm6.w = mass[j + 6u];
            float4 pm7 = pos_in[j + 7u]; pm7.w = mass[j + 7u];

            acc0 = nbody_accum_one(acc0, ri, pm0, eps2);
            acc1 = nbody_accum_one(acc1, ri, pm1, eps2);
            acc2 = nbody_accum_one(acc2, ri, pm2, eps2);
            acc3 = nbody_accum_one(acc3, ri, pm3, eps2);
            acc0 = nbody_accum_one(acc0, ri, pm4, eps2);
            acc1 = nbody_accum_one(acc1, ri, pm5, eps2);
            acc2 = nbody_accum_one(acc2, ri, pm6, eps2);
            acc3 = nbody_accum_one(acc3, ri, pm7, eps2);
        }

        for (; j + 3u < N; j += 4u) {
            float4 pm0 = pos_in[j + 0u]; pm0.w = mass[j + 0u];
            float4 pm1 = pos_in[j + 1u]; pm1.w = mass[j + 1u];
            float4 pm2 = pos_in[j + 2u]; pm2.w = mass[j + 2u];
            float4 pm3 = pos_in[j + 3u]; pm3.w = mass[j + 3u];

            acc0 = nbody_accum_one(acc0, ri, pm0, eps2);
            acc1 = nbody_accum_one(acc1, ri, pm1, eps2);
            acc2 = nbody_accum_one(acc2, ri, pm2, eps2);
            acc3 = nbody_accum_one(acc3, ri, pm3, eps2);
        }

        for (; j < N; ++j) {
            float4 pm = pos_in[j];
            pm.w = mass[j];
            acc0 = nbody_accum_one(acc0, ri, pm, eps2);
        }

        float3 acc = (acc0 + acc1) + (acc2 + acc3);
        float3 v_new = vi + acc * gdt;
        float3 r_new = ri + v_new * dt;

        pos_out[i] = float4(r_new, 0.0f);
        vel_out[i] = float4(v_new, 0.0f);
        return;
    }

    // Uniform whole-threadgroup exit before any barriers.
    const uint group_start = i - tid;
    if (group_start >= N) return;

    const bool active = (i < N);

    float3 ri = float3(0.0f);
    float3 vi = float3(0.0f);
    if (active) {
        ri = pos_in[i].xyz;
        vi = vel_in[i].xyz;
    }

    float3 acc0 = float3(0.0f);
    float3 acc1 = float3(0.0f);
    float3 acc2 = float3(0.0f);
    float3 acc3 = float3(0.0f);

    for (uint base = 0; base < N; base += NBODY_TILE_SIZE) {
        for (uint l = tid; l < NBODY_TILE_SIZE; l += tg_size) {
            uint src = base + l;
            if (src < N) {
                float4 pm = pos_in[src];
                pm.w = mass[src];
                tile[l] = pm;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            uint remaining = N - base;
            uint count = (remaining < NBODY_TILE_SIZE) ? remaining : NBODY_TILE_SIZE;

            uint k = 0;
            for (; k + 7u < count; k += 8u) {
                float4 pm0 = tile[k + 0u];
                float4 pm1 = tile[k + 1u];
                float4 pm2 = tile[k + 2u];
                float4 pm3 = tile[k + 3u];
                float4 pm4 = tile[k + 4u];
                float4 pm5 = tile[k + 5u];
                float4 pm6 = tile[k + 6u];
                float4 pm7 = tile[k + 7u];

                acc0 = nbody_accum_one(acc0, ri, pm0, eps2);
                acc1 = nbody_accum_one(acc1, ri, pm1, eps2);
                acc2 = nbody_accum_one(acc2, ri, pm2, eps2);
                acc3 = nbody_accum_one(acc3, ri, pm3, eps2);
                acc0 = nbody_accum_one(acc0, ri, pm4, eps2);
                acc1 = nbody_accum_one(acc1, ri, pm5, eps2);
                acc2 = nbody_accum_one(acc2, ri, pm6, eps2);
                acc3 = nbody_accum_one(acc3, ri, pm7, eps2);
            }

            for (; k + 3u < count; k += 4u) {
                acc0 = nbody_accum_one(acc0, ri, tile[k + 0u], eps2);
                acc1 = nbody_accum_one(acc1, ri, tile[k + 1u], eps2);
                acc2 = nbody_accum_one(acc2, ri, tile[k + 2u], eps2);
                acc3 = nbody_accum_one(acc3, ri, tile[k + 3u], eps2);
            }

            for (; k < count; ++k) {
                acc0 = nbody_accum_one(acc0, ri, tile[k], eps2);
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

Incumbent result:
            256_10: correct, 0.41 ms, 32.0 GFLOPS (0.7% of 4500 GFLOPS)
           1024_10: correct, 0.75 ms, 280.0 GFLOPS (6.2% of 4500 GFLOPS)
           2048_10: correct, 1.19 ms, 705.5 GFLOPS (15.7% of 4500 GFLOPS)
  score (gmean of fraction): 0.0411

## History

- iter  0: compile=OK | correct=True | score=0.02032597394032279
- iter  1: compile=FAIL | correct=False | score=N/A
- iter  2: compile=OK | correct=True | score=0.041078093951235344
- iter  3: compile=FAIL | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
