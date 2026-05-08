## Task: heat2d

2D heat equation with a 5-point stencil:
  u_new[i,j] = u[i,j] + alpha * (u[i-1,j] + u[i+1,j]
                                 + u[i,j-1] + u[i,j+1]
                                 - 4 u[i,j])
Dirichlet BC: boundary cells stay at their initial value. Row-major float32 storage of shape (NY, NX) — i indexes columns (fast axis), j indexes rows. Stable for alpha <= 0.25; we use alpha = 0.20 below the limit. The host runs the kernel for n_steps iterations with two buffers ping-ponged each call.

## Required kernel signature(s)

```
kernel void heat_step(device const float *u_in  [[buffer(0)]],
                      device       float *u_out [[buffer(1)]],
                      constant uint      &NX    [[buffer(2)]],
                      constant uint      &NY    [[buffer(3)]],
                      constant float     &alpha [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]]);

Grid is dispatched 2-D as `threadsPerGrid = (NX, NY)`, one thread per output cell — guard with `if (i >= NX || j >= NY) return;`. Each thread MUST update exactly one cell; the host will not shrink the dispatch if you process multiple cells per thread, so extra threads just idle. Boundary cells (i==0, j==0, i==NX-1, j==NY-1) must copy u_in -> u_out unchanged.
```

## Your previous attempt

```metal
#include <metal_stdlib>
using namespace metal;

kernel void heat_step(device const float *u_in  [[buffer(0)]],
                      device       float *u_out [[buffer(1)]],
                      constant uint      &NX    [[buffer(2)]],
                      constant uint      &NY    [[buffer(3)]],
                      constant float     &alpha [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]],
                      uint2 lid [[thread_position_in_threadgroup]],
                      uint2 tpg [[threads_per_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]]) {
    const uint i  = gid.x;
    const uint j  = gid.y;
    const uint nx = NX;
    const uint ny = NY;

    if (i >= nx || j >= ny) return;

    const uint idx = j * nx + i;
    const float c = u_in[idx];

    // All in-grid lanes load c and execute these shuffles before boundary lanes
    // copy out, so interior cells can safely read boundary neighbors via lanes.
    const uint lane_u = uint(lane);

    const float sh_l = simd_shuffle(c, ushort((lane_u > 0u)  ? (lane_u - 1u) : 0u));
    const float sh_r = simd_shuffle(c, ushort((lane_u < 31u) ? (lane_u + 1u) : 31u));

    const uint row_w = tpg.x;
    float sh_up = c;
    float sh_dn = c;

    // If a SIMDgroup contains multiple threadgroup rows, vertical neighbors may
    // also be in registers.  Otherwise these remain unused and we fall back.
    if (row_w < 32u) {
        const uint up_lane = (lane_u >= row_w) ? (lane_u - row_w) : lane_u;
        const uint dn_lane = ((lane_u + row_w) < 32u) ? (lane_u + row_w) : lane_u;
        sh_up = simd_shuffle(c, ushort(up_lane));
        sh_dn = simd_shuffle(c, ushort(dn_lane));
    }

    if (i == 0u || j == 0u || i + 1u >= nx || j + 1u >= ny) {
        u_out[idx] = c;
        return;
    }

    float l;
    if (lid.x != 0u && lane_u != 0u) {
        l = sh_l;
    } else {
        l = u_in[idx - 1u];
    }

    float r;
    if ((lid.x + 1u) < row_w && lane_u != 31u) {
        r = sh_r;
    } else {
        r = u_in[idx + 1u];
    }

    float upv;
    if (lid.y != 0u && lane_u >= row_w) {
        upv = sh_up;
    } else {
        upv = u_in[idx - nx];
    }

    float dnv;
    if ((lid.y + 1u) < tpg.y && (lane_u + row_w) < 32u) {
        dnv = sh_dn;
    } else {
        dnv = u_in[idx + nx];
    }

    u_out[idx] = c + alpha * (l + r + upv + dnv - 4.0f * c);
}
```

Result of previous attempt:
        256x256_50: correct, 0.64 ms, 41.3 GB/s (effective, 8 B/cell) (20.6% of 200 GB/s)
       512x512_100: correct, 3.72 ms, 56.4 GB/s (effective, 8 B/cell) (28.2% of 200 GB/s)
      1024x1024_50: correct, 5.02 ms, 83.6 GB/s (effective, 8 B/cell) (41.8% of 200 GB/s)
  score (gmean of fraction): 0.2897

## Current best (incumbent)

```metal
// Naive seed kernel for the 2D heat equation, one timestep, 5-point stencil.
//
//     u_new[i,j] = u[i,j] + alpha * (u[i-1,j] + u[i+1,j]
//                                    + u[i,j-1] + u[i,j+1]
//                                    - 4 u[i,j])
//
// Dirichlet boundary conditions: edge cells stay at their initial value.
//
// Buffer layout:
//   buffer 0: const float* u_in   (NX * NY, row-major)
//   buffer 1: device float* u_out (NX * NY, row-major)
//   buffer 2: const uint& NX
//   buffer 3: const uint& NY
//   buffer 4: const float& alpha  (alpha = D * dt / dx^2 in [0, 0.25])

#include <metal_stdlib>
using namespace metal;

kernel void heat_step(device const float *u_in  [[buffer(0)]],
                      device       float *u_out [[buffer(1)]],
                      constant uint      &NX    [[buffer(2)]],
                      constant uint      &NY    [[buffer(3)]],
                      constant float     &alpha [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;  // column
    uint j = gid.y;  // row
    if (i >= NX || j >= NY) return;

    if (i == 0 || j == 0 || i == NX - 1 || j == NY - 1) {
        // Dirichlet: copy boundary value through unchanged.
        u_out[j * NX + i] = u_in[j * NX + i];
        return;
    }

    float c = u_in[j * NX + i];
    float l = u_in[j * NX + (i - 1)];
    float r = u_in[j * NX + (i + 1)];
    float d = u_in[(j - 1) * NX + i];
    float u = u_in[(j + 1) * NX + i];
    u_out[j * NX + i] = c + alpha * (l + r + d + u - 4.0f * c);
}
```

Incumbent result:
        256x256_50: correct, 0.37 ms, 70.1 GB/s (effective, 8 B/cell) (35.1% of 200 GB/s)
       512x512_100: correct, 1.26 ms, 165.8 GB/s (effective, 8 B/cell) (82.9% of 200 GB/s)
      1024x1024_50: correct, 1.87 ms, 224.3 GB/s (effective, 8 B/cell) (112.2% of 200 GB/s)
  score (gmean of fraction): 0.6883

## History

- iter  2: compile=OK | correct=True | score=0.3463215725221932
- iter  3: compile=OK | correct=True | score=0.2961192957141125
- iter  4: compile=OK | correct=True | score=0.3900575907602331
- iter  5: compile=OK | correct=True | score=0.3070785537375746
- iter  6: compile=OK | correct=True | score=0.4193805336288794
- iter  7: compile=OK | correct=True | score=0.4273516006535969
- iter  8: compile=OK | correct=True | score=0.442587634390779
- iter  9: compile=OK | correct=True | score=0.2896515294140095

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
