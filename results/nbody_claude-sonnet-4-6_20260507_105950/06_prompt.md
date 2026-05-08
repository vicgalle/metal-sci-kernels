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

constant uint TILE = 256;

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
                       uint i  [[thread_position_in_grid]],
                       uint li [[thread_position_in_threadgroup]])
{
    // Tiled O(N^2) all-pairs: each tile of TILE bodies is cooperatively
    // loaded into threadgroup memory so each body is read once per tile
    // and reused TILE times, improving memory bandwidth utilization.
    // NaN fix: out-of-bounds padding uses a far-away zero-mass body so
    // distance is large and contribution is exactly zero (no rsqrt(0)).

    threadgroup float4 tg_pos[TILE];
    threadgroup float  tg_mass[TILE];

    // Load own position/velocity (or dummy values for out-of-bounds threads)
    float3 ri   = (i < N) ? pos_in[i].xyz  : float3(0.0f);
    float3 vi   = (i < N) ? vel_in[i].xyz  : float3(0.0f);

    float3 a    = float3(0.0f);
    float  eps2 = eps * eps;
    float  Gval = G;

    uint num_tiles = (N + TILE - 1) / TILE;

    for (uint t = 0; t < num_tiles; ++t) {
        uint src = t * TILE + li;

        // Cooperative load — pad with a distant zero-mass body to avoid
        // rsqrt(0) and ensure zero contribution safely.
        if (src < N) {
            tg_pos[li]  = pos_in[src];
            tg_mass[li] = mass[src];
        } else {
            // Place padding body 1e18 units away (effectively zero force,
            // no division by zero even if eps==0)
            tg_pos[li]  = float4(1e18f, 0.0f, 0.0f, 0.0f);
            tg_mass[li] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // How many entries are valid in this tile
        uint tile_start = t * TILE;
        uint tile_end   = min(tile_start + TILE, N);
        uint tile_count = tile_end - tile_start;

        // Full 8-wide unrolled inner loop
        uint jj = 0;
        for (; jj + 8u <= tile_count; jj += 8u) {
            float3 d0 = tg_pos[jj  ].xyz - ri;
            float3 d1 = tg_pos[jj+1].xyz - ri;
            float3 d2 = tg_pos[jj+2].xyz - ri;
            float3 d3 = tg_pos[jj+3].xyz - ri;
            float3 d4 = tg_pos[jj+4].xyz - ri;
            float3 d5 = tg_pos[jj+5].xyz - ri;
            float3 d6 = tg_pos[jj+6].xyz - ri;
            float3 d7 = tg_pos[jj+7].xyz - ri;

            float r0sq = dot(d0,d0) + eps2;
            float r1sq = dot(d1,d1) + eps2;
            float r2sq = dot(d2,d2) + eps2;
            float r3sq = dot(d3,d3) + eps2;
            float r4sq = dot(d4,d4) + eps2;
            float r5sq = dot(d5,d5) + eps2;
            float r6sq = dot(d6,d6) + eps2;
            float r7sq = dot(d7,d7) + eps2;

            // inv_r3 = (r^2)^{-3/2} = rsqrt(r^6) = rsqrt(r2)^3
            float s0 = rsqrt(r0sq); s0 = s0 * s0 * s0;
            float s1 = rsqrt(r1sq); s1 = s1 * s1 * s1;
            float s2 = rsqrt(r2sq); s2 = s2 * s2 * s2;
            float s3 = rsqrt(r3sq); s3 = s3 * s3 * s3;
            float s4 = rsqrt(r4sq); s4 = s4 * s4 * s4;
            float s5 = rsqrt(r5sq); s5 = s5 * s5 * s5;
            float s6 = rsqrt(r6sq); s6 = s6 * s6 * s6;
            float s7 = rsqrt(r7sq); s7 = s7 * s7 * s7;

            a += d0 * (tg_mass[jj  ] * s0);
            a += d1 * (tg_mass[jj+1] * s1);
            a += d2 * (tg_mass[jj+2] * s2);
            a += d3 * (tg_mass[jj+3] * s3);
            a += d4 * (tg_mass[jj+4] * s4);
            a += d5 * (tg_mass[jj+5] * s5);
            a += d6 * (tg_mass[jj+6] * s6);
            a += d7 * (tg_mass[jj+7] * s7);
        }
        // Scalar tail for partial last tile
        for (; jj < tile_count; ++jj) {
            float3 d  = tg_pos[jj].xyz - ri;
            float  r2 = dot(d,d) + eps2;
            float  s  = rsqrt(r2); s = s * s * s;
            a += d * (tg_mass[jj] * s);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (i >= N) return;

    a *= Gval;

    float3 v_new = vi + a * dt;
    float3 r_new = ri + v_new * dt;

    pos_out[i] = float4(r_new, pos_in[i].w);
    vel_out[i] = float4(v_new, vel_in[i].w);
}
```

Result of previous attempt:
            256_10: INCORRECT (max_abs_pos=1.076e-02, tol=0.0019946529269218444)
  fail_reason: correctness failed at size 256_10: max_abs_pos=1.076e-02

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
- iter  4: compile=OK | correct=False | score=N/A
- iter  5: compile=OK | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
