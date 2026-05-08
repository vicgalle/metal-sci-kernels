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
#define HALO 1
#define SMEM (TILE + 2*HALO)

kernel void heat_step(device const float *u_in  [[buffer(0)]],
                      device       float *u_out [[buffer(1)]],
                      constant uint      &NX    [[buffer(2)]],
                      constant uint      &NY    [[buffer(3)]],
                      constant float     &alpha [[buffer(4)]],
                      uint2 gid  [[thread_position_in_grid]],
                      uint2 lid  [[thread_position_in_threadgroup]],
                      uint2 tgid [[threadgroup_position_in_grid]])
[[max_total_threads_per_threadgroup(256)]]
{
    threadgroup float s[SMEM][SMEM];

    uint i = gid.x;
    uint j = gid.y;

    int li = int(lid.x);
    int lj = int(lid.y);

    // Tile origin in global coords.
    int base_i = int(tgid.x) * TILE;
    int base_j = int(tgid.y) * TILE;

    // Cooperative load: each of the 16x16 = 256 threads loads multiple
    // elements of the 18x18 = 324 shared region.
    int tid = lj * TILE + li;
    for (int k = tid; k < SMEM * SMEM; k += TILE * TILE) {
        int sy = k / SMEM;
        int sx = k - sy * SMEM;
        int gx = base_i + sx - HALO;
        int gy = base_j + sy - HALO;
        // Clamp to valid range so we never read OOB. Boundary cells
        // are not used for stencil updates of interior cells anyway,
        // and clamping keeps memory accesses safe.
        gx = clamp(gx, 0, int(NX) - 1);
        gy = clamp(gy, 0, int(NY) - 1);
        s[sy][sx] = u_in[gy * int(NX) + gx];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (i >= NX || j >= NY) return;

    uint idx = j * NX + i;

    if (i == 0 || j == 0 || i == NX - 1 || j == NY - 1) {
        u_out[idx] = u_in[idx];
        return;
    }

    int sx = li + HALO;
    int sy = lj + HALO;

    float c  = s[sy    ][sx    ];
    float l  = s[sy    ][sx - 1];
    float r  = s[sy    ][sx + 1];
    float dn = s[sy - 1][sx    ];
    float up = s[sy + 1][sx    ];

    u_out[idx] = c + alpha * ((l + r) + (dn + up) - 4.0f * c);
}
```

Result of previous attempt:
  COMPILE FAILED: Error Domain=MTLLibraryErrorDomain Code=3 "program_source:16:3: error: 'max_total_threads_per_threadgroup' attribute cannot be applied to types
[[max_total_threads_per_threadgroup(256)]]
  ^
" UserInfo={NSLocalizedDescription=program_source:16:3: error: 'max_total_threads_per_threadgroup' attribute cannot be applied to types
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
- iter  5: compile=OK | correct=True | score=0.42994973909414436
- iter  6: compile=FAIL | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
