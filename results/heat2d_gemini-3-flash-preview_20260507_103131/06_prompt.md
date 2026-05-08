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
                      uint2 tsize [[threads_per_threadgroup]]) {
    const uint i = gid.x;
    const uint j = gid.y;

    // Tile storage for threadgroup-level data reuse.
    // Supports threadgroups up to 32x32 (though 256 total threads is recommended).
    threadgroup float tile[34][34];

    const uint ti = lid.x;
    const uint tj = lid.y;

    if (i < NX && j < NY) {
        const uint idx = j * NX + i;

        // Cooperative load: each thread fetches its central value.
        tile[tj + 1][ti + 1] = u_in[idx];

        // Load the 1-pixel apron (Left, Right, Up, Down). 
        // Corners are not required for a 5-point stencil.
        if (ti == 0 && i > 0) {
            tile[tj + 1][0] = u_in[idx - 1];
        }
        if (ti == tsize.x - 1 && i < NX - 1) {
            tile[tj + 1][ti + 2] = u_in[idx + 1];
        }
        if (tj == 0 && j > 0) {
            tile[0][ti + 1] = u_in[idx - NX];
        }
        if (tj == tsize.y - 1 && j < NY - 1) {
            tile[tj + 2][ti + 1] = u_in[idx + NX];
        }

        // Synchronize threads to ensure the tile is fully loaded.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Dirichlet Boundary Condition: check if the cell is on the grid edge.
        if (i > 0 && j > 0 && i < NX - 1 && j < NY - 1) {
            const float c = tile[tj + 1][ti + 1];
            const float l = tile[tj + 1][ti];
            const float r = tile[tj + 1][ti + 2];
            const float u = tile[tj][ti + 1];
            const float d = tile[tj + 2][ti + 1];

            // Discrete Laplacian and update using optimized math.
            // u_new = u + alpha * (L + R + U + D - 4*u)
            const float laplacian = (l + r + u + d) - 4.0f * c;
            u_out[idx] = fma(alpha, laplacian, c);
        } else {
            // Edges remain unchanged (Dirichlet BC).
            u_out[idx] = tile[tj + 1][ti + 1];
        }
    }
}
```

Result of previous attempt:
        256x256_50: correct, 1.04 ms, 25.2 GB/s (effective, 8 B/cell) (12.6% of 200 GB/s)
       512x512_100: correct, 2.70 ms, 77.7 GB/s (effective, 8 B/cell) (38.9% of 200 GB/s)
      1024x1024_50: correct, 3.94 ms, 106.4 GB/s (effective, 8 B/cell) (53.2% of 200 GB/s)
  score (gmean of fraction): 0.2965

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
        256x256_50: correct, 0.37 ms, 71.1 GB/s (effective, 8 B/cell) (35.5% of 200 GB/s)
       512x512_100: correct, 1.31 ms, 160.2 GB/s (effective, 8 B/cell) (80.1% of 200 GB/s)
      1024x1024_50: correct, 1.88 ms, 223.0 GB/s (effective, 8 B/cell) (111.5% of 200 GB/s)
  score (gmean of fraction): 0.6821

## History

- iter  0: compile=OK | correct=True | score=0.6821190507104802
- iter  1: compile=OK | correct=True | score=0.2782111491198901
- iter  2: compile=OK | correct=True | score=0.4042673852636761
- iter  3: compile=OK | correct=True | score=0.5446965872681632
- iter  4: compile=OK | correct=True | score=0.4120114505573336
- iter  5: compile=OK | correct=True | score=0.2964550182965342

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
