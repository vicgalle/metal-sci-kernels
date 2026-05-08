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

#define TILE_W 32
#define TILE_H 8

kernel void heat_step(device const float *u_in  [[buffer(0)]],
                      device       float *u_out [[buffer(1)]],
                      constant uint      &NX    [[buffer(2)]],
                      constant uint      &NY    [[buffer(3)]],
                      constant float     &alpha [[buffer(4)]],
                      uint2 gid   [[thread_position_in_grid]],
                      uint2 tid   [[thread_position_in_threadgroup]])
    __attribute__((max_total_threads_per_threadgroup(TILE_W * TILE_H)))
{
    // Shared memory tile with 1-cell halo: (TILE_H+2) x (TILE_W+2)
    threadgroup float smem[(TILE_H + 2) * (TILE_W + 2)];
    const uint sw = TILE_W + 2;

    const uint i = gid.x;
    const uint j = gid.y;

    const uint lx = tid.x + 1;
    const uint ly = tid.y + 1;

    // Safe (clamped) global indices for this thread
    const uint si = min(i, NX - 1);
    const uint sj = min(j, NY - 1);

    // Load center into smem
    smem[ly * sw + lx] = u_in[sj * NX + si];

    // Left halo (tid.x == 0 loads the column to the left)
    if (tid.x == 0) {
        uint hx = (i > 0) ? (si - 1) : 0;
        smem[ly * sw + 0] = u_in[sj * NX + hx];
    }
    // Right halo
    if (tid.x == TILE_W - 1) {
        uint hx = min(si + 1, NX - 1);
        smem[ly * sw + (TILE_W + 1)] = u_in[sj * NX + hx];
    }
    // Top halo (j-1 direction; j is row, so j-1 = "above" in memory)
    if (tid.y == 0) {
        uint hy = (j > 0) ? (sj - 1) : 0;
        smem[0 * sw + lx] = u_in[hy * NX + si];
    }
    // Bottom halo
    if (tid.y == TILE_H - 1) {
        uint hy = min(sj + 1, NY - 1);
        smem[(TILE_H + 1) * sw + lx] = u_in[hy * NX + si];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Guard after barrier so all threads participate in the barrier
    if (i >= NX || j >= NY) return;

    float center = smem[ly * sw + lx];

    if (i == 0 || j == 0 || i == NX - 1 || j == NY - 1) {
        u_out[j * NX + i] = center;
        return;
    }

    float l = smem[ly       * sw + (lx - 1)];
    float r = smem[ly       * sw + (lx + 1)];
    float up = smem[(ly - 1) * sw + lx];
    float dn = smem[(ly + 1) * sw + lx];

    u_out[j * NX + i] = center + alpha * (l + r + up + dn - 4.0f * center);
}
```

Result of previous attempt:
        256x256_50: INCORRECT (max_abs=nan, tol=0.00010578201711177826)
  fail_reason: correctness failed at size 256x256_50: max_abs=nan

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
        256x256_50: correct, 0.72 ms, 36.5 GB/s (effective, 8 B/cell) (18.3% of 200 GB/s)
       512x512_100: correct, 1.26 ms, 166.7 GB/s (effective, 8 B/cell) (83.3% of 200 GB/s)
      1024x1024_50: correct, 1.89 ms, 222.1 GB/s (effective, 8 B/cell) (111.0% of 200 GB/s)
  score (gmean of fraction): 0.5528

## History

- iter  1: compile=FAIL | correct=False | score=N/A
- iter  2: compile=FAIL | correct=False | score=N/A
- iter  3: compile=FAIL | correct=False | score=N/A
- iter  4: compile=FAIL | correct=False | score=N/A
- iter  5: compile=FAIL | correct=False | score=N/A
- iter  6: compile=FAIL | correct=False | score=N/A
- iter  7: compile=OK | correct=False | score=N/A
- iter  8: compile=OK | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
