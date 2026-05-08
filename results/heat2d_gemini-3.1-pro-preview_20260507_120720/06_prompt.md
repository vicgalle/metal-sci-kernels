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
                      uint2 tsize [[threads_per_threadgroup]]) {
    
    // Allocate shared memory for up to 1024 threads.
    // Max tile size occurs at a 1024x1 threadgroup: (1024+2)*(1+2) = 3078 elements.
    threadgroup float tile[3078];
    
    uint nx = NX;
    uint ny = NY;
    uint i = gid.x;
    uint j = gid.y;
    
    bool valid = (i < nx && j < ny);
    
    uint stride = tsize.x + 2;
    #define TILE_IDX(y, x) ((y) * stride + (x))
    
    // 1. Center load (also kept in register for reuse)
    float c_val = valid ? u_in[j * nx + i] : 0.0f;
    tile[TILE_IDX(tid.y + 1, tid.x + 1)] = c_val;
    
    // 2. Halo loads (clamps strictly bounds-checked to avoid page faults)
    if (tid.y == 0) {
        uint y = (j > 0) ? j - 1 : 0;
        tile[TILE_IDX(0, tid.x + 1)] = valid ? u_in[y * nx + i] : 0.0f;
    }
    if (tid.y == tsize.y - 1) {
        uint y = (j < ny - 1) ? j + 1 : j;
        tile[TILE_IDX(tsize.y + 1, tid.x + 1)] = valid ? u_in[y * nx + i] : 0.0f;
    }
    
    if (tid.x == 0) {
        uint x = (i > 0) ? i - 1 : 0;
        tile[TILE_IDX(tid.y + 1, 0)] = valid ? u_in[j * nx + x] : 0.0f;
    }
    if (tid.x == tsize.x - 1) {
        uint x = (i < nx - 1) ? i + 1 : i;
        tile[TILE_IDX(tid.y + 1, tsize.x + 1)] = valid ? u_in[j * nx + x] : 0.0f;
    }
    
    // 3. Synchronize threadgroup memory visibility
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // 4. Out-of-bounds threads safely exit after participating in the barrier
    if (!valid) return;
    
    uint idx = j * nx + i;
    
    // 5. Dirichlet boundaries bypass the stencil update
    if (i == 0 || j == 0 || i == nx - 1 || j == ny - 1) {
        u_out[idx] = c_val;
        return;
    }
    
    // 6. Stencil computation for interior cells
    float l = tile[TILE_IDX(tid.y + 1, tid.x)];
    float r = tile[TILE_IDX(tid.y + 1, tid.x + 2)];
    float t = tile[TILE_IDX(tid.y,     tid.x + 1)];
    float b = tile[TILE_IDX(tid.y + 2, tid.x + 1)];
    
    float a = alpha;
    u_out[idx] = c_val + a * (l + r + t + b - 4.0f * c_val);
    
    #undef TILE_IDX
}
```

Result of previous attempt:
        256x256_50: correct, 1.23 ms, 21.3 GB/s (effective, 8 B/cell) (10.6% of 200 GB/s)
       512x512_100: correct, 3.42 ms, 61.4 GB/s (effective, 8 B/cell) (30.7% of 200 GB/s)
      1024x1024_50: correct, 4.20 ms, 100.0 GB/s (effective, 8 B/cell) (50.0% of 200 GB/s)
  score (gmean of fraction): 0.2537

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
        256x256_50: correct, 0.74 ms, 35.6 GB/s (effective, 8 B/cell) (17.8% of 200 GB/s)
       512x512_100: correct, 2.28 ms, 92.0 GB/s (effective, 8 B/cell) (46.0% of 200 GB/s)
      1024x1024_50: correct, 1.88 ms, 222.8 GB/s (effective, 8 B/cell) (111.4% of 200 GB/s)
  score (gmean of fraction): 0.4504

## History

- iter  0: compile=OK | correct=True | score=0.4503535203778002
- iter  1: compile=OK | correct=True | score=0.3597485724594344
- iter  2: compile=OK | correct=True | score=0.4015504678097701
- iter  3: compile=OK | correct=True | score=0.39531045899529954
- iter  4: compile=OK | correct=True | score=0.41408714517370904
- iter  5: compile=OK | correct=True | score=0.25366881587902923

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
