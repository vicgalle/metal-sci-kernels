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

#define NBODY_TILE_BODIES 1024u
#define NBODY_TILE_QUADS  256u

static inline float3 nbody_accum_one(float3 acc, float3 ri, float4 pm, float eps2) {
    float3 d = pm.xyz - ri;
    float r2 = dot(d, d) + eps2;
    float inv_r = rsqrt(r2);
    float inv_r2 = inv_r * inv_r;
    float s = pm.w * inv_r * inv_r2;
    return acc + d * s;
}

static inline float nbody_hsum4(float4 v) {
    return (v.x + v.y) + (v.z + v.w);
}

#define NBODY_LOAD_TILE_EXACT(BASE_)                                           \
do {                                                                           \
    for (uint q__ = tid; q__ < NBODY_TILE_QUADS; q__ += tg_size) {             \
        const uint src__ = (BASE_) + (q__ << 2);                               \
        const float4 p0__ = pos_in[src__ + 0u];                                \
        const float4 p1__ = pos_in[src__ + 1u];                                \
        const float4 p2__ = pos_in[src__ + 2u];                                \
        const float4 p3__ = pos_in[src__ + 3u];                                \
        tile_x[q__] = float4(p0__.x, p1__.x, p2__.x, p3__.x);                  \
        tile_y[q__] = float4(p0__.y, p1__.y, p2__.y, p3__.y);                  \
        tile_z[q__] = float4(p0__.z, p1__.z, p2__.z, p3__.z);                  \
        tile_m[q__] = float4(mass[src__ + 0u], mass[src__ + 1u],               \
                             mass[src__ + 2u], mass[src__ + 3u]);              \
    }                                                                          \
} while (false)

#define NBODY_ACCUM_VEC(AX_, AY_, AZ_, Q_)                                     \
do {                                                                           \
    const float4 dx__ = tile_x[(Q_)] - float4(ri.x);                           \
    const float4 dy__ = tile_y[(Q_)] - float4(ri.y);                           \
    const float4 dz__ = tile_z[(Q_)] - float4(ri.z);                           \
    const float4 r2__ = fma(dx__, dx__, fma(dy__, dy__,                        \
                            fma(dz__, dz__, float4(eps2))));                   \
    const float4 inv__ = rsqrt(r2__);                                          \
    const float4 inv2__ = inv__ * inv__;                                       \
    const float4 s__ = tile_m[(Q_)] * inv__ * inv2__;                          \
    (AX_) = fma(dx__, s__, (AX_));                                             \
    (AY_) = fma(dy__, s__, (AY_));                                             \
    (AZ_) = fma(dz__, s__, (AZ_));                                             \
} while (false)

#define NBODY_COMPUTE_TILE_EXACT()                                             \
do {                                                                           \
    if (active) {                                                              \
        for (uint q__ = 0u; q__ < NBODY_TILE_QUADS; q__ += 2u) {               \
            NBODY_ACCUM_VEC(ax0, ay0, az0, q__ + 0u);                          \
            NBODY_ACCUM_VEC(ax1, ay1, az1, q__ + 1u);                          \
        }                                                                      \
    }                                                                          \
} while (false)

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
    threadgroup float4 tile_x[NBODY_TILE_QUADS];
    threadgroup float4 tile_y[NBODY_TILE_QUADS];
    threadgroup float4 tile_z[NBODY_TILE_QUADS];
    threadgroup float4 tile_m[NBODY_TILE_QUADS];

    const float eps2 = eps * eps;
    const float gdt = G * dt;

    // Small problem path: no barriers, preserving low overhead for N=256/512.
    if (N <= 512u) {
        if (i >= N) return;

        const float3 ri = pos_in[i].xyz;
        const float3 vi = vel_in[i].xyz;

        float3 acc0 = float3(0.0f);
        float3 acc1 = float3(0.0f);
        float3 acc2 = float3(0.0f);
        float3 acc3 = float3(0.0f);

        uint j = 0u;
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

    const uint group_start = i - tid;
    if (group_start >= N) return;

    const bool active = (i < N);

    float3 ri = float3(0.0f);
    float3 vi = float3(0.0f);
    if (active) {
        ri = pos_in[i].xyz;
        vi = vel_in[i].xyz;
    }

    float4 ax0 = float4(0.0f), ay0 = float4(0.0f), az0 = float4(0.0f);
    float4 ax1 = float4(0.0f), ay1 = float4(0.0f), az1 = float4(0.0f);

    // Exact hot path for benchmark N=1024: one full 1024-body tile.
    if (N == 1024u) {
        NBODY_LOAD_TILE_EXACT(0u);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        NBODY_COMPUTE_TILE_EXACT();

        if (active) {
            const float3 acc = float3(nbody_hsum4(ax0 + ax1),
                                      nbody_hsum4(ay0 + ay1),
                                      nbody_hsum4(az0 + az1));
            const float3 v_new = vi + acc * gdt;
            const float3 r_new = ri + v_new * dt;
            pos_out[i] = float4(r_new, 0.0f);
            vel_out[i] = float4(v_new, 0.0f);
        }
        return;
    }

    // Exact hot path for benchmark N=2048: two full 1024-body tiles.
    if (N == 2048u) {
        NBODY_LOAD_TILE_EXACT(0u);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        NBODY_COMPUTE_TILE_EXACT();

        threadgroup_barrier(mem_flags::mem_none);

        NBODY_LOAD_TILE_EXACT(1024u);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        NBODY_COMPUTE_TILE_EXACT();

        if (active) {
            const float3 acc = float3(nbody_hsum4(ax0 + ax1),
                                      nbody_hsum4(ay0 + ay1),
                                      nbody_hsum4(az0 + az1));
            const float3 v_new = vi + acc * gdt;
            const float3 r_new = ri + v_new * dt;
            pos_out[i] = float4(r_new, 0.0f);
            vel_out[i] = float4(v_new, 0.0f);
        }
        return;
    }

    // Correct generic fallback for other N > 512.
    for (uint base = 0u; base < N; base += NBODY_TILE_BODIES) {
        const uint remaining = N - base;
        const uint bodies = (remaining < NBODY_TILE_BODIES) ? remaining : NBODY_TILE_BODIES;
        const uint qcount = (bodies + 3u) >> 2;

        for (uint q = tid; q < qcount; q += tg_size) {
            const uint src = base + (q << 2);

            float4 tx = float4(0.0f);
            float4 ty = float4(0.0f);
            float4 tz = float4(0.0f);
            float4 tm = float4(0.0f);

            if (src + 0u < N) {
                const float4 p = pos_in[src + 0u];
                tx.x = p.x; ty.x = p.y; tz.x = p.z; tm.x = mass[src + 0u];
            }
            if (src + 1u < N) {
                const float4 p = pos_in[src + 1u];
                tx.y = p.x; ty.y = p.y; tz.y = p.z; tm.y = mass[src + 1u];
            }
            if (src + 2u < N) {
                const float4 p = pos_in[src + 2u];
                tx.z = p.x; ty.z = p.y; tz.z = p.z; tm.z = mass[src + 2u];
            }
            if (src + 3u < N) {
                const float4 p = pos_in[src + 3u];
                tx.w = p.x; ty.w = p.y; tz.w = p.z; tm.w = mass[src + 3u];
            }

            tile_x[q] = tx;
            tile_y[q] = ty;
            tile_z[q] = tz;
            tile_m[q] = tm;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            uint q = 0u;
            for (; q + 1u < qcount; q += 2u) {
                NBODY_ACCUM_VEC(ax0, ay0, az0, q + 0u);
                NBODY_ACCUM_VEC(ax1, ay1, az1, q + 1u);
            }
            if (q < qcount) {
                NBODY_ACCUM_VEC(ax0, ay0, az0, q);
            }
        }

        if (base + NBODY_TILE_BODIES < N) {
            threadgroup_barrier(mem_flags::mem_none);
        }
    }

    if (active) {
        const float3 acc = float3(nbody_hsum4(ax0 + ax1),
                                  nbody_hsum4(ay0 + ay1),
                                  nbody_hsum4(az0 + az1));
        const float3 v_new = vi + acc * gdt;
        const float3 r_new = ri + v_new * dt;
        pos_out[i] = float4(r_new, 0.0f);
        vel_out[i] = float4(v_new, 0.0f);
    }
}

#undef NBODY_COMPUTE_TILE_EXACT
#undef NBODY_ACCUM_VEC
#undef NBODY_LOAD_TILE_EXACT
#undef NBODY_TILE_QUADS
#undef NBODY_TILE_BODIES
```

Result of previous attempt:
            256_10: correct, 0.41 ms, 32.1 GFLOPS (0.7% of 4500 GFLOPS)
           1024_10: correct, 1.14 ms, 184.1 GFLOPS (4.1% of 4500 GFLOPS)
           2048_10: correct, 0.93 ms, 906.5 GFLOPS (20.1% of 4500 GFLOPS)
  score (gmean of fraction): 0.0389

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
- iter  4: compile=OK | correct=True | score=0.03355391271159659
- iter  5: compile=OK | correct=True | score=0.0388665923677914

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
