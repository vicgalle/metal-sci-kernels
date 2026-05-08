Fixed the compile failure by using only scalar 1-D threadgroup built-ins and removing the misplaced `max_total_threads_per_threadgroup` attribute. The kernel now uses cooperative threadgroup tiling for larger `N`, packing position and mass into shared memory so each source body is fetched once per threadgroup tile instead of once per target thread. The inner loop is manually unrolled with multiple accumulators to improve instruction-level parallelism and hide `rsqrt` latency; small `N` keeps a direct unrolled path to avoid barrier overhead.

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