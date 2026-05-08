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
                      uint2 tid [[thread_position_in_threadgroup]],
                      uint2 tpg [[threads_per_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    uint i = gid.x;
    uint j = gid.y;

    if (i >= NX || j >= NY) return;

    uint idx = j * NX + i;
    float c = u_in[idx];

    // All valid lanes, including boundary lanes, participate so neighboring
    // interior lanes can read boundary values through SIMD shuffles.
    float sh_left  = simd_shuffle_up(c, ushort(1));
    float sh_right = simd_shuffle_down(c, ushort(1));

    float sh_up = 0.0f;
    float sh_down = 0.0f;
    bool can_shuffle_vertical = (tpg.x < 32u);
    if (can_shuffle_vertical) {
        ushort dx = ushort(tpg.x);
        sh_up   = simd_shuffle_up(c, dx);
        sh_down = simd_shuffle_down(c, dx);
    }

    bool interior = ((i - 1u) < (NX - 2u)) && ((j - 1u) < (NY - 2u));

    if (!interior) {
        u_out[idx] = c;
        return;
    }

    float l;
    if (tid.x != 0u && lane != 0u) {
        l = sh_left;
    } else {
        l = u_in[idx - 1u];
    }

    float r;
    if ((tid.x + 1u < tpg.x) && lane != 31u) {
        r = sh_right;
    } else {
        r = u_in[idx + 1u];
    }

    float upv;
    if (can_shuffle_vertical && tid.y != 0u && lane >= tpg.x) {
        upv = sh_up;
    } else {
        upv = u_in[idx - NX];
    }

    float downv;
    if (can_shuffle_vertical && (tid.y + 1u < tpg.y) && (lane + tpg.x < 32u)) {
        downv = sh_down;
    } else {
        downv = u_in[idx + NX];
    }

    float lap = l + r + upv + downv - 4.0f * c;
    u_out[idx] = c + alpha * lap;
}
```

Result of previous attempt:
        256x256_50: correct, 0.88 ms, 29.7 GB/s (effective, 8 B/cell) (14.9% of 200 GB/s)
       512x512_100: correct, 2.09 ms, 100.1 GB/s (effective, 8 B/cell) (50.1% of 200 GB/s)
      1024x1024_50: correct, 3.22 ms, 130.4 GB/s (effective, 8 B/cell) (65.2% of 200 GB/s)
  score (gmean of fraction): 0.3647

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

- iter  0: compile=OK | correct=True | score=0.6883214595661554
- iter  1: compile=OK | correct=True | score=0.36474589028668897

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
