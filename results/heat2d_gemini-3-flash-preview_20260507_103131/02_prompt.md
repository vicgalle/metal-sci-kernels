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
                      uint2 lsize [[threads_per_threadgroup]]) {
    const int i = (int)gid.x;
    const int j = (int)gid.y;
    const int nx = (int)NX;
    const int ny = (int)NY;
    const int lx = (int)lid.x;
    const int ly = (int)lid.y;
    const int sw = (int)lsize.x;
    const int sh = (int)lsize.y;
    const int stride = sw + 2;

    // Threadgroup memory to store the tile plus a 1-pixel halo.
    // 4096 floats = 16KB, which is well within the 32KB limit on Apple Silicon.
    // This size supports any configuration where (sw+2)*(sh+2) <= 4096.
    threadgroup float tile[4096];

    const bool in_bounds = (i < nx && j < ny);
    const int idx = j * nx + i;
    const int tile_idx = (ly + 1) * stride + (lx + 1);

    if (in_bounds) {
        // Every thread loads its central value into the tile.
        const float center_val = u_in[idx];
        tile[tile_idx] = center_val;

        // Cooperative halo loading: only threads on the edge of the threadgroup 
        // load neighbor values that are outside the current block.
        if (lx == 0 && i > 0) 
            tile[tile_idx - 1] = u_in[idx - 1];
        if (lx == sw - 1 && i < nx - 1) 
            tile[tile_idx + 1] = u_in[idx + 1];
        if (ly == 0 && j > 0) 
            tile[tile_idx - stride] = u_in[idx - nx];
        if (ly == sh - 1 && j < ny - 1) 
            tile[tile_idx + stride] = u_in[idx + nx];
    }

    // Synchronize to ensure all threads in the group have finished loading.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!in_bounds) return;

    // Dirichlet Boundary Conditions: Edge cells copy initial value and return.
    if (i == 0 || j == 0 || i == nx - 1 || j == ny - 1) {
        u_out[idx] = tile[tile_idx];
    } else {
        // Access 5-point stencil values from the high-speed shared tile.
        const float c = tile[tile_idx];
        const float l = tile[tile_idx - 1];
        const float r = tile[tile_idx + 1];
        const float u = tile[tile_idx - stride];
        const float d = tile[tile_idx + stride];

        // u_new = u + alpha * Laplacian
        u_out[idx] = c + alpha * (l + r + u + d - 4.0f * c);
    }
}
```

Result of previous attempt:
        256x256_50: correct, 1.15 ms, 22.8 GB/s (effective, 8 B/cell) (11.4% of 200 GB/s)
       512x512_100: correct, 2.90 ms, 72.3 GB/s (effective, 8 B/cell) (36.2% of 200 GB/s)
      1024x1024_50: correct, 4.02 ms, 104.3 GB/s (effective, 8 B/cell) (52.2% of 200 GB/s)
  score (gmean of fraction): 0.2782

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

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
