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

#define TILE 16

kernel void __attribute__((max_total_threads_per_threadgroup(TILE*TILE)))
heat_step(device const float *u_in  [[buffer(0)]],
          device       float *u_out [[buffer(1)]],
          constant uint      &NX    [[buffer(2)]],
          constant uint      &NY    [[buffer(3)]],
          constant float     &alpha [[buffer(4)]],
          uint2 gid  [[thread_position_in_grid]],
          uint2 lid  [[thread_position_in_threadgroup]],
          uint2 tgid [[threadgroup_position_in_grid]])
{
    threadgroup float tile[TILE + 2][TILE + 2];

    uint i = gid.x;
    uint j = gid.y;

    int li = int(lid.x) + 1;
    int lj = int(lid.y) + 1;

    // Origin of this tile in global coords.
    int gi0 = int(tgid.x) * TILE;
    int gj0 = int(tgid.y) * TILE;

    // Center load (clamped so OOB threads still get a sane value; we won't
    // write OOB).
    uint ci = min(i, NX - 1);
    uint cj = min(j, NY - 1);
    tile[lj][li] = u_in[cj * NX + ci];

    // Halo loads. Each edge of the threadgroup loads its corresponding
    // halo strip; corners are handled by the corner threads.
    if (lid.x == 0) {
        int gi = gi0 - 1;
        uint si = uint(max(gi, 0));
        uint sj = min(uint(gj0 + int(lid.y)), NY - 1);
        tile[lj][0] = u_in[sj * NX + si];
    }
    if (lid.x == TILE - 1) {
        int gi = gi0 + TILE;
        uint si = uint(min(gi, int(NX) - 1));
        uint sj = min(uint(gj0 + int(lid.y)), NY - 1);
        tile[lj][TILE + 1] = u_in[sj * NX + si];
    }
    if (lid.y == 0) {
        int gj = gj0 - 1;
        uint sj = uint(max(gj, 0));
        uint si = min(uint(gi0 + int(lid.x)), NX - 1);
        tile[0][li] = u_in[sj * NX + si];
    }
    if (lid.y == TILE - 1) {
        int gj = gj0 + TILE;
        uint sj = uint(min(gj, int(NY) - 1));
        uint si = min(uint(gi0 + int(lid.x)), NX - 1);
        tile[TILE + 1][li] = u_in[sj * NX + si];
    }
    // Corners
    if (lid.x == 0 && lid.y == 0) {
        int gi = gi0 - 1, gj = gj0 - 1;
        uint si = uint(max(gi, 0));
        uint sj = uint(max(gj, 0));
        tile[0][0] = u_in[sj * NX + si];
    }
    if (lid.x == TILE - 1 && lid.y == 0) {
        int gi = gi0 + TILE, gj = gj0 - 1;
        uint si = uint(min(gi, int(NX) - 1));
        uint sj = uint(max(gj, 0));
        tile[0][TILE + 1] = u_in[sj * NX + si];
    }
    if (lid.x == 0 && lid.y == TILE - 1) {
        int gi = gi0 - 1, gj = gj0 + TILE;
        uint si = uint(max(gi, 0));
        uint sj = uint(min(gj, int(NY) - 1));
        tile[TILE + 1][0] = u_in[sj * NX + si];
    }
    if (lid.x == TILE - 1 && lid.y == TILE - 1) {
        int gi = gi0 + TILE, gj = gj0 + TILE;
        uint si = uint(min(gi, int(NX) - 1));
        uint sj = uint(min(gj, int(NY) - 1));
        tile[TILE + 1][TILE + 1] = u_in[sj * NX + si];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (i >= NX || j >= NY) return;

    uint idx = j * NX + i;
    float c = tile[lj][li];

    if (i == 0 || j == 0 || i == NX - 1 || j == NY - 1) {
        u_out[idx] = c;
        return;
    }

    float l = tile[lj][li - 1];
    float r = tile[lj][li + 1];
    float d = tile[lj - 1][li];
    float uu = tile[lj + 1][li];

    u_out[idx] = c + alpha * ((l + r) + (d + uu) - 4.0f * c);
}
```

Result of previous attempt:
        256x256_50: correct, 1.31 ms, 20.1 GB/s (effective, 8 B/cell) (10.0% of 200 GB/s)
       512x512_100: correct, 3.41 ms, 61.5 GB/s (effective, 8 B/cell) (30.8% of 200 GB/s)
      1024x1024_50: correct, 4.26 ms, 98.5 GB/s (effective, 8 B/cell) (49.2% of 200 GB/s)
  score (gmean of fraction): 0.2478

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

- iter  0: compile=OK | correct=True | score=0.6087461540822107
- iter  1: compile=FAIL | correct=False | score=N/A
- iter  2: compile=OK | correct=True | score=0.32409516943637545
- iter  3: compile=FAIL | correct=False | score=N/A
- iter  4: compile=OK | correct=True | score=0.24777726205461026

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
