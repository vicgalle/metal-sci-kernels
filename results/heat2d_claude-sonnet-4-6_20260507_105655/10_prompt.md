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
#define SW (TILE_W + 2)
#define SH (TILE_H + 2)

kernel void heat_step(device const float *u_in  [[buffer(0)]],
                      device       float *u_out [[buffer(1)]],
                      constant uint      &NX    [[buffer(2)]],
                      constant uint      &NY    [[buffer(3)]],
                      constant float     &alpha [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]],
                      uint2 tid [[thread_position_in_threadgroup]],
                      uint2 tgid [[threadgroup_position_in_grid]])
    __attribute__((max_total_threads_per_threadgroup(TILE_W * TILE_H)))
{
    threadgroup float smem[SH * SW];

    // smem indices for this thread's center cell (with 1-cell border)
    const uint lx = tid.x + 1;
    const uint ly = tid.y + 1;

    // Global indices for this thread
    const uint i = gid.x;
    const uint j = gid.y;

    // Clamp to valid range for loading (guards against out-of-bounds reads)
    const uint ci = min(i, NX - 1);
    const uint cj = min(j, NY - 1);

    // Load center
    smem[ly * SW + lx] = u_in[cj * NX + ci];

    // Halo loads: use actual neighboring global coords, clamped to grid
    // Left halo column (tid.x == 0)
    if (tid.x == 0) {
        uint hx = (i > 0) ? (i - 1) : 0;
        hx = min(hx, NX - 1);
        smem[ly * SW + 0] = u_in[cj * NX + hx];
    }
    // Right halo column (tid.x == TILE_W - 1, but last tile may be smaller)
    // Use tid.x == TILE_W-1 OR actual last thread in threadgroup
    if (tid.x == TILE_W - 1 || i == NX - 1) {
        uint hx = min(i + 1, NX - 1);
        smem[ly * SW + (lx + 1)] = u_in[cj * NX + hx];
    }
    // Top halo row (tid.y == 0)
    if (tid.y == 0) {
        uint hy = (j > 0) ? (j - 1) : 0;
        hy = min(hy, NY - 1);
        smem[0 * SW + lx] = u_in[hy * NX + ci];
    }
    // Bottom halo row
    if (tid.y == TILE_H - 1 || j == NY - 1) {
        uint hy = min(j + 1, NY - 1);
        smem[(ly + 1) * SW + lx] = u_in[hy * NX + ci];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Guard after barrier
    if (i >= NX || j >= NY) return;

    float center = smem[ly * SW + lx];

    if (i == 0 || j == 0 || i == NX - 1 || j == NY - 1) {
        u_out[j * NX + i] = center;
        return;
    }

    float left  = smem[ly       * SW + (lx - 1)];
    float right = smem[ly       * SW + (lx + 1)];
    float up    = smem[(ly - 1) * SW + lx];
    float down  = smem[(ly + 1) * SW + lx];

    u_out[j * NX + i] = center + alpha * (left + right + up + down - 4.0f * center);
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

- iter  2: compile=FAIL | correct=False | score=N/A
- iter  3: compile=FAIL | correct=False | score=N/A
- iter  4: compile=FAIL | correct=False | score=N/A
- iter  5: compile=FAIL | correct=False | score=N/A
- iter  6: compile=FAIL | correct=False | score=N/A
- iter  7: compile=OK | correct=False | score=N/A
- iter  8: compile=OK | correct=False | score=N/A
- iter  9: compile=OK | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
