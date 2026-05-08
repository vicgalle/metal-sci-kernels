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

#define NBODY_TILE_SIZE 512u
#define NBODY_SOA_QUADS 128u

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

#define NBODY_DEVICE_ACCUM4(J_)                                             \
do {                                                                        \
    const uint j__ = (J_);                                                   \
    float4 pm0__ = pos_in[j__ + 0u]; pm0__.w = mass[j__ + 0u];               \
    float4 pm1__ = pos_in[j__ + 1u]; pm1__.w = mass[j__ + 1u];               \
    float4 pm2__ = pos_in[j__ + 2u]; pm2__.w = mass[j__ + 2u];               \
    float4 pm3__ = pos_in[j__ + 3u]; pm3__.w = mass[j__ + 3u];               \
    acc0 = nbody_accum_one(acc0, ri, pm0__, eps2);                           \
    acc1 = nbody_accum_one(acc1, ri, pm1__, eps2);                           \
    acc2 = nbody_accum_one(acc2, ri, pm2__, eps2);                           \
    acc3 = nbody_accum_one(acc3, ri, pm3__, eps2);                           \
} while (false)

#define NBODY_DEVICE_ACCUM8(J_)                                             \
do {                                                                        \
    const uint j__ = (J_);                                                   \
    float4 pm0__ = pos_in[j__ + 0u]; pm0__.w = mass[j__ + 0u];               \
    float4 pm1__ = pos_in[j__ + 1u]; pm1__.w = mass[j__ + 1u];               \
    float4 pm2__ = pos_in[j__ + 2u]; pm2__.w = mass[j__ + 2u];               \
    float4 pm3__ = pos_in[j__ + 3u]; pm3__.w = mass[j__ + 3u];               \
    float4 pm4__ = pos_in[j__ + 4u]; pm4__.w = mass[j__ + 4u];               \
    float4 pm5__ = pos_in[j__ + 5u]; pm5__.w = mass[j__ + 5u];               \
    float4 pm6__ = pos_in[j__ + 6u]; pm6__.w = mass[j__ + 6u];               \
    float4 pm7__ = pos_in[j__ + 7u]; pm7__.w = mass[j__ + 7u];               \
    acc0 = nbody_accum_one(acc0, ri, pm0__, eps2);                           \
    acc1 = nbody_accum_one(acc1, ri, pm1__, eps2);                           \
    acc2 = nbody_accum_one(acc2, ri, pm2__, eps2);                           \
    acc3 = nbody_accum_one(acc3, ri, pm3__, eps2);                           \
    acc0 = nbody_accum_one(acc0, ri, pm4__, eps2);                           \
    acc1 = nbody_accum_one(acc1, ri, pm5__, eps2);                           \
    acc2 = nbody_accum_one(acc2, ri, pm6__, eps2);                           \
    acc3 = nbody_accum_one(acc3, ri, pm7__, eps2);                           \
} while (false)

#define NBODY_TILE_ACCUM4(K_)                                               \
do {                                                                        \
    const uint k__ = (K_);                                                   \
    acc0 = nbody_accum_one(acc0, ri, shared[k__ + 0u], eps2);                \
    acc1 = nbody_accum_one(acc1, ri, shared[k__ + 1u], eps2);                \
    acc2 = nbody_accum_one(acc2, ri, shared[k__ + 2u], eps2);                \
    acc3 = nbody_accum_one(acc3, ri, shared[k__ + 3u], eps2);                \
} while (false)

#define NBODY_TILE_ACCUM8(K_)                                               \
do {                                                                        \
    const uint k__ = (K_);                                                   \
    const float4 pm0__ = shared[k__ + 0u];                                   \
    const float4 pm1__ = shared[k__ + 1u];                                   \
    const float4 pm2__ = shared[k__ + 2u];                                   \
    const float4 pm3__ = shared[k__ + 3u];                                   \
    const float4 pm4__ = shared[k__ + 4u];                                   \
    const float4 pm5__ = shared[k__ + 5u];                                   \
    const float4 pm6__ = shared[k__ + 6u];                                   \
    const float4 pm7__ = shared[k__ + 7u];                                   \
    acc0 = nbody_accum_one(acc0, ri, pm0__, eps2);                           \
    acc1 = nbody_accum_one(acc1, ri, pm1__, eps2);                           \
    acc2 = nbody_accum_one(acc2, ri, pm2__, eps2);                           \
    acc3 = nbody_accum_one(acc3, ri, pm3__, eps2);                           \
    acc0 = nbody_accum_one(acc0, ri, pm4__, eps2);                           \
    acc1 = nbody_accum_one(acc1, ri, pm5__, eps2);                           \
    acc2 = nbody_accum_one(acc2, ri, pm6__, eps2);                           \
    acc3 = nbody_accum_one(acc3, ri, pm7__, eps2);                           \
} while (false)

#define NBODY_LOAD_AOS512_EXACT(BASE_)                                      \
do {                                                                        \
    for (uint l__ = tid; l__ < NBODY_TILE_SIZE; l__ += tg_size) {            \
        const uint src__ = (BASE_) + l__;                                    \
        float4 pm__ = pos_in[src__];                                         \
        pm__.w = mass[src__];                                                \
        shared[l__] = pm__;                                                  \
    }                                                                        \
} while (false)

#define NBODY_LOAD_AOS512_BOUNDS(BASE_)                                     \
do {                                                                        \
    for (uint l__ = tid; l__ < NBODY_TILE_SIZE; l__ += tg_size) {            \
        const uint src__ = (BASE_) + l__;                                    \
        if (src__ < N) {                                                     \
            float4 pm__ = pos_in[src__];                                     \
            pm__.w = mass[src__];                                            \
            shared[l__] = pm__;                                              \
        }                                                                    \
    }                                                                        \
} while (false)

#define NBODY_LOAD_SOA512_EXACT(BASE_)                                      \
do {                                                                        \
    for (uint q__ = tid; q__ < NBODY_SOA_QUADS; q__ += tg_size) {            \
        const uint src__ = (BASE_) + (q__ << 2);                             \
        const float4 p0__ = pos_in[src__ + 0u];                              \
        const float4 p1__ = pos_in[src__ + 1u];                              \
        const float4 p2__ = pos_in[src__ + 2u];                              \
        const float4 p3__ = pos_in[src__ + 3u];                              \
        shared[q__]                         = float4(p0__.x, p1__.x, p2__.x, p3__.x); \
        shared[NBODY_SOA_QUADS + q__]       = float4(p0__.y, p1__.y, p2__.y, p3__.y); \
        shared[NBODY_SOA_QUADS * 2u + q__]  = float4(p0__.z, p1__.z, p2__.z, p3__.z); \
        shared[NBODY_SOA_QUADS * 3u + q__]  = float4(mass[src__ + 0u], mass[src__ + 1u], \
                                                      mass[src__ + 2u], mass[src__ + 3u]); \
    }                                                                        \
} while (false)

#define NBODY_ACCUM_SOA(AX_, AY_, AZ_, Q_)                                  \
do {                                                                        \
    const float4 dx__ = shared[(Q_)] - float4(ri.x);                         \
    const float4 dy__ = shared[NBODY_SOA_QUADS + (Q_)] - float4(ri.y);        \
    const float4 dz__ = shared[NBODY_SOA_QUADS * 2u + (Q_)] - float4(ri.z);   \
    const float4 r2__ = fma(dx__, dx__, fma(dy__, dy__, fma(dz__, dz__, float4(eps2)))); \
    const float4 inv__ = rsqrt(r2__);                                        \
    const float4 inv2__ = inv__ * inv__;                                     \
    const float4 s__ = shared[NBODY_SOA_QUADS * 3u + (Q_)] * inv__ * inv2__;  \
    (AX_) = fma(dx__, s__, (AX_));                                           \
    (AY_) = fma(dy__, s__, (AY_));                                           \
    (AZ_) = fma(dz__, s__, (AZ_));                                           \
} while (false)

#define NBODY_COMPUTE_SOA512_EXACT()                                        \
do {                                                                        \
    if (active) {                                                            \
        for (uint q__ = 0u; q__ < NBODY_SOA_QUADS; q__ += 4u) {              \
            NBODY_ACCUM_SOA(ax0, ay0, az0, q__ + 0u);                        \
            NBODY_ACCUM_SOA(ax1, ay1, az1, q__ + 1u);                        \
            NBODY_ACCUM_SOA(ax0, ay0, az0, q__ + 2u);                        \
            NBODY_ACCUM_SOA(ax1, ay1, az1, q__ + 3u);                        \
        }                                                                    \
    }                                                                        \
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
    threadgroup float4 shared[NBODY_TILE_SIZE];

    const float eps2 = eps * eps;
    const float gdt = G * dt;

    if (N == 256u) {
        if (i >= 256u) return;

        const float3 ri = pos_in[i].xyz;
        const float3 vi = vel_in[i].xyz;

        float3 acc0 = float3(0.0f);
        float3 acc1 = float3(0.0f);
        float3 acc2 = float3(0.0f);
        float3 acc3 = float3(0.0f);

        for (uint j = 0u; j < 256u; j += 16u) {
            NBODY_DEVICE_ACCUM8(j);
            NBODY_DEVICE_ACCUM8(j + 8u);
        }

        const float3 acc = (acc0 + acc1) + (acc2 + acc3);
        const float3 v_new = vi + acc * gdt;
        const float3 r_new = ri + v_new * dt;

        pos_out[i] = float4(r_new, 0.0f);
        vel_out[i] = float4(v_new, 0.0f);
        return;
    }

    if (N <= NBODY_TILE_SIZE) {
        if (i >= N) return;

        const float3 ri = pos_in[i].xyz;
        const float3 vi = vel_in[i].xyz;

        float3 acc0 = float3(0.0f);
        float3 acc1 = float3(0.0f);
        float3 acc2 = float3(0.0f);
        float3 acc3 = float3(0.0f);

        uint j = 0u;
        for (; j + 15u < N; j += 16u) {
            NBODY_DEVICE_ACCUM8(j);
            NBODY_DEVICE_ACCUM8(j + 8u);
        }
        for (; j + 7u < N; j += 8u) {
            NBODY_DEVICE_ACCUM8(j);
        }
        for (; j + 3u < N; j += 4u) {
            NBODY_DEVICE_ACCUM4(j);
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

    if (N == 1024u) {
        float3 acc0 = float3(0.0f);
        float3 acc1 = float3(0.0f);
        float3 acc2 = float3(0.0f);
        float3 acc3 = float3(0.0f);

        NBODY_LOAD_AOS512_EXACT(0u);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            for (uint k = 0u; k < NBODY_TILE_SIZE; k += 16u) {
                NBODY_TILE_ACCUM8(k);
                NBODY_TILE_ACCUM8(k + 8u);
            }
        }

        threadgroup_barrier(mem_flags::mem_none);

        NBODY_LOAD_AOS512_EXACT(512u);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            for (uint k = 0u; k < NBODY_TILE_SIZE; k += 16u) {
                NBODY_TILE_ACCUM8(k);
                NBODY_TILE_ACCUM8(k + 8u);
            }

            const float3 acc = (acc0 + acc1) + (acc2 + acc3);
            const float3 v_new = vi + acc * gdt;
            const float3 r_new = ri + v_new * dt;

            pos_out[i] = float4(r_new, 0.0f);
            vel_out[i] = float4(v_new, 0.0f);
        }
        return;
    }

    if (N == 2048u) {
        float4 ax0 = float4(0.0f), ay0 = float4(0.0f), az0 = float4(0.0f);
        float4 ax1 = float4(0.0f), ay1 = float4(0.0f), az1 = float4(0.0f);

        NBODY_LOAD_SOA512_EXACT(0u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        NBODY_COMPUTE_SOA512_EXACT();

        threadgroup_barrier(mem_flags::mem_none);

        NBODY_LOAD_SOA512_EXACT(512u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        NBODY_COMPUTE_SOA512_EXACT();

        threadgroup_barrier(mem_flags::mem_none);

        NBODY_LOAD_SOA512_EXACT(1024u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        NBODY_COMPUTE_SOA512_EXACT();

        threadgroup_barrier(mem_flags::mem_none);

        NBODY_LOAD_SOA512_EXACT(1536u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        NBODY_COMPUTE_SOA512_EXACT();

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

    float3 acc0 = float3(0.0f);
    float3 acc1 = float3(0.0f);
    float3 acc2 = float3(0.0f);
    float3 acc3 = float3(0.0f);

    for (uint base = 0u; base < N; base += NBODY_TILE_SIZE) {
        NBODY_LOAD_AOS512_BOUNDS(base);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            const uint remaining = N - base;
            const uint count = (remaining < NBODY_TILE_SIZE) ? remaining : NBODY_TILE_SIZE;

            uint k = 0u;
            for (; k + 15u < count; k += 16u) {
                NBODY_TILE_ACCUM8(k);
                NBODY_TILE_ACCUM8(k + 8u);
            }
            for (; k + 7u < count; k += 8u) {
                NBODY_TILE_ACCUM8(k);
            }
            for (; k + 3u < count; k += 4u) {
                NBODY_TILE_ACCUM4(k);
            }
            for (; k < count; ++k) {
                acc0 = nbody_accum_one(acc0, ri, shared[k], eps2);
            }
        }

        if (base + NBODY_TILE_SIZE < N) {
            threadgroup_barrier(mem_flags::mem_none);
        }
    }

    if (active) {
        const float3 acc = (acc0 + acc1) + (acc2 + acc3);
        const float3 v_new = vi + acc * gdt;
        const float3 r_new = ri + v_new * dt;

        pos_out[i] = float4(r_new, 0.0f);
        vel_out[i] = float4(v_new, 0.0f);
    }
}

#undef NBODY_COMPUTE_SOA512_EXACT
#undef NBODY_ACCUM_SOA
#undef NBODY_LOAD_SOA512_EXACT
#undef NBODY_LOAD_AOS512_BOUNDS
#undef NBODY_LOAD_AOS512_EXACT
#undef NBODY_TILE_ACCUM8
#undef NBODY_TILE_ACCUM4
#undef NBODY_DEVICE_ACCUM8
#undef NBODY_DEVICE_ACCUM4
#undef NBODY_SOA_QUADS
#undef NBODY_TILE_SIZE
```

Result of previous attempt:
            256_10: correct, 0.38 ms, 34.9 GFLOPS (0.8% of 4500 GFLOPS)
           1024_10: correct, 0.67 ms, 312.3 GFLOPS (6.9% of 4500 GFLOPS)
           2048_10: correct, 1.14 ms, 737.0 GFLOPS (16.4% of 4500 GFLOPS)
  score (gmean of fraction): 0.0445

## History

- iter  1: compile=FAIL | correct=False | score=N/A
- iter  2: compile=OK | correct=True | score=0.041078093951235344
- iter  3: compile=FAIL | correct=False | score=N/A
- iter  4: compile=OK | correct=True | score=0.03355391271159659
- iter  5: compile=OK | correct=True | score=0.0388665923677914
- iter  6: compile=OK | correct=True | score=0.038879049773156704
- iter  7: compile=OK | correct=True | score=0.039324793488981755
- iter  8: compile=OK | correct=True | score=0.04449155020695442

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
