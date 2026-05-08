I use a SIMDgroup-broadcast path for 32-aligned problem sizes, with one SIMDgroup per threadgroup and lane-cooperative source loads. This removes threadgroup-memory barriers, improves scheduling granularity for small/medium N, and keeps all source reads coalesced while each lane still computes exactly one body update. A scalar FMA accumulator helper improves instruction scheduling and reduces vector temporary overhead.

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