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
                      uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x;
    const uint j = gid.y;

    // Bounds check to handle dispatches larger than the grid
    if (i >= NX || j >= NY) return;

    const uint idx = j * NX + i;

    // Dirichlet Boundary Conditions: Edge cells stay at their initial value.
    // Grouping i and j checks to allow the compiler to optimize branch logic.
    if (i == 0 || i == NX - 1 || j == 0 || j == NY - 1) {
        u_out[idx] = u_in[idx];
        return;
    }

    // Stencil loads: center and its 4 neighbors.
    // Using relative offsets to avoid re-calculating (j-1)*NX and (j+1)*NX.
    const float c = u_in[idx];
    const float l = u_in[idx - 1];
    const float r = u_in[idx + 1];
    const float d = u_in[idx - NX]; // Row above
    const float u = u_in[idx + NX]; // Row below

    // Discrete Laplacian calculation: (L + R + D + U - 4*C)
    // Structure additions to encourage Instruction-Level Parallelism (ILP).
    const float neighbors_sum = (l + r) + (d + u);
    const float laplacian = neighbors_sum - 4.0f * c;

    // Heat Equation Step: u_new = u + alpha * Laplacian
    // Use fma for precision and to potentially combine operations into one cycle.
    u_out[idx] = fma(alpha, laplacian, c);
}
```

Result of previous attempt:
        256x256_50: correct, 0.77 ms, 34.2 GB/s (effective, 8 B/cell) (17.1% of 200 GB/s)
       512x512_100: correct, 3.34 ms, 62.8 GB/s (effective, 8 B/cell) (31.4% of 200 GB/s)
      1024x1024_50: correct, 2.39 ms, 175.8 GB/s (effective, 8 B/cell) (87.9% of 200 GB/s)
  score (gmean of fraction): 0.3612

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

- iter  1: compile=OK | correct=True | score=0.2782111491198901
- iter  2: compile=OK | correct=True | score=0.4042673852636761
- iter  3: compile=OK | correct=True | score=0.5446965872681632
- iter  4: compile=OK | correct=True | score=0.4120114505573336
- iter  5: compile=OK | correct=True | score=0.2964550182965342
- iter  6: compile=OK | correct=True | score=0.5293643127805301
- iter  7: compile=OK | correct=True | score=0.3422860181034194
- iter  8: compile=OK | correct=True | score=0.36124429835282074

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
