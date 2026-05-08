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

constant uint TILE_W = 16;
constant uint TILE_H = 16;

kernel void heat_step(device const float *u_in  [[buffer(0)]],
                      device       float *u_out [[buffer(1)]],
                      constant uint      &NX    [[buffer(2)]],
                      constant uint      &NY    [[buffer(3)]],
                      constant float     &alpha [[buffer(4)]],
                      uint2 gid  [[thread_position_in_grid]],
                      uint2 tid  [[thread_position_in_threadgroup]],
                      uint2 tgid [[threadgroup_position_in_grid]])
    [[max_total_threads_per_threadgroup(256)]]
{
    // Shared tile with 1-cell halo on each side: (TILE_W+2) x (TILE_H+2)
    threadgroup float smem[(TILE_H + 2) * (TILE_W + 2)];
    const uint sw = TILE_W + 2;

    uint i = gid.x;
    uint j = gid.y;

    uint lx = tid.x + 1;  // local x with halo offset
    uint ly = tid.y + 1;  // local y with halo offset

    // Clamp coordinates for safe OOB loads (grid may be padded)
    uint ci = min(i, NX - 1);
    uint cj = min(j, NY - 1);

    // Load interior of tile
    smem[ly * sw + lx] = u_in[cj * NX + ci];

    // Left halo
    if (tid.x == 0) {
        uint hi = (i > 0) ? (i - 1) : 0;
        uint hci = min(hi, NX - 1);
        smem[ly * sw + 0] = u_in[cj * NX + hci];
    }
    // Right halo
    if (tid.x == TILE_W - 1) {
        uint hi = i + 1;
        uint hci = min(hi, NX - 1);
        smem[ly * sw + (TILE_W + 1)] = u_in[cj * NX + hci];
    }
    // Top halo (j-1)
    if (tid.y == 0) {
        uint hj = (j > 0) ? (j - 1) : 0;
        uint hcj = min(hj, NY - 1);
        smem[0 * sw + lx] = u_in[hcj * NX + ci];
    }
    // Bottom halo (j+1)
    if (tid.y == TILE_H - 1) {
        uint hj = j + 1;
        uint hcj = min(hj, NY - 1);
        smem[(TILE_H + 1) * sw + lx] = u_in[hcj * NX + ci];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (i >= NX || j >= NY) return;

    // Boundary cells: copy unchanged (Dirichlet BC)
    if (i == 0 || j == 0 || i == NX - 1 || j == NY - 1) {
        u_out[j * NX + i] = smem[ly * sw + lx];
        return;
    }

    float c = smem[ly       * sw + lx    ];
    float l = smem[ly       * sw + lx - 1];
    float r = smem[ly       * sw + lx + 1];
    float t = smem[(ly - 1) * sw + lx    ];
    float b = smem[(ly + 1) * sw + lx    ];

    u_out[j * NX + i] = c + alpha * (l + r + t + b - 4.0f * c);
}
```

Result of previous attempt:
  COMPILE FAILED: Error Domain=MTLLibraryErrorDomain Code=3 "program_source:15:7: error: 'max_total_threads_per_threadgroup' attribute cannot be applied to types
    [[max_total_threads_per_threadgroup(256)]]
      ^
" UserInfo={NSLocalizedDescription=program_source:15:7: error: 'max_total_threads_per_threadgroup' attribute cannot be applied to types
    [[max_total_threads_per_threadgroup(256)]]
      ^
}

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

- iter  0: compile=OK | correct=True | score=0.5528189743916646
- iter  1: compile=FAIL | correct=False | score=N/A
- iter  2: compile=FAIL | correct=False | score=N/A
- iter  3: compile=FAIL | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
