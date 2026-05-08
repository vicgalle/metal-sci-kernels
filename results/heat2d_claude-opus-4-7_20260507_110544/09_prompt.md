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

[[max_total_threads_per_threadgroup(256)]]
kernel void heat_step(device const float *u_in  [[buffer(0)]],
                      device       float *u_out [[buffer(1)]],
                      constant uint      &NX    [[buffer(2)]],
                      constant uint      &NY    [[buffer(3)]],
                      constant float     &alpha [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]])
{
    uint i = gid.x;
    uint j = gid.y;
    if (i >= NX || j >= NY) return;

    uint nx = NX;
    uint idx = j * nx + i;

    if (i == 0 || j == 0 || i == nx - 1 || j == NY - 1) {
        u_out[idx] = u_in[idx];
        return;
    }

    device const float *row  = u_in + idx;
    float c = row[0];
    float l = row[-1];
    float r = row[1];
    float d = row[-(int)nx];
    float up = row[nx];

    u_out[idx] = c + alpha * ((l + r) + (d + up) - 4.0f * c);
}
```

Result of previous attempt:
        256x256_50: correct, 0.79 ms, 33.0 GB/s (effective, 8 B/cell) (16.5% of 200 GB/s)
       512x512_100: correct, 1.87 ms, 112.4 GB/s (effective, 8 B/cell) (56.2% of 200 GB/s)
      1024x1024_50: correct, 2.62 ms, 160.2 GB/s (effective, 8 B/cell) (80.1% of 200 GB/s)
  score (gmean of fraction): 0.4204

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
        256x256_50: correct, 0.35 ms, 74.3 GB/s (effective, 8 B/cell) (37.2% of 200 GB/s)
       512x512_100: correct, 1.94 ms, 108.2 GB/s (effective, 8 B/cell) (54.1% of 200 GB/s)
      1024x1024_50: correct, 1.87 ms, 224.3 GB/s (effective, 8 B/cell) (112.2% of 200 GB/s)
  score (gmean of fraction): 0.6087

## History

- iter  1: compile=FAIL | correct=False | score=N/A
- iter  2: compile=OK | correct=True | score=0.32409516943637545
- iter  3: compile=FAIL | correct=False | score=N/A
- iter  4: compile=OK | correct=True | score=0.24777726205461026
- iter  5: compile=OK | correct=True | score=0.42994973909414436
- iter  6: compile=FAIL | correct=False | score=N/A
- iter  7: compile=OK | correct=True | score=0.4384797002006627
- iter  8: compile=OK | correct=True | score=0.42038475856265256

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
