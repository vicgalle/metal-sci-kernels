## Task: wave3d

3D acoustic wave equation with a 7-point spatial Laplacian and second-order leapfrog time integration:
  u_next[i,j,k] = 2 u_curr[i,j,k] - u_prev[i,j,k]
                + alpha * ( u_curr[i-1,j,k] + u_curr[i+1,j,k]
                          + u_curr[i,j-1,k] + u_curr[i,j+1,k]
                          + u_curr[i,j,k-1] + u_curr[i,j,k+1]
                          - 6 u_curr[i,j,k] )
alpha = (c * dt / dx)^2; the host uses alpha = 0.18, comfortably below the 3D CFL limit of 1/3. Dirichlet BC: every face cell (i==0, j==0, k==0, i==NX-1, j==NY-1, k==NZ-1) MUST copy u_curr -> u_next unchanged.

Storage is row-major float32 of shape (NZ, NY, NX) — i is the fast (x) axis, j the middle (y) axis, k the slow (z) axis. Linear index: idx = (k * NY + j) * NX + i. The host triple-buffers across n_steps, rotating (prev, curr, next) bindings each call; all dispatches share one command buffer for accurate end-to-end GPU timing. Initial state has u_prev = u_curr (zero initial velocity in time).

## Required kernel signature(s)

```
kernel void wave_step(device const float *u_prev [[buffer(0)]],
                      device const float *u_curr [[buffer(1)]],
                      device       float *u_next [[buffer(2)]],
                      constant uint      &NX     [[buffer(3)]],
                      constant uint      &NY     [[buffer(4)]],
                      constant uint      &NZ     [[buffer(5)]],
                      constant float     &alpha  [[buffer(6)]],
                      uint3 gid [[thread_position_in_grid]]);

Grid is dispatched 3-D as `threadsPerGrid = (NX, NY, NZ)`, one thread per output cell — guard with `if (i >= NX || j >= NY || k >= NZ) return;`. Each thread MUST update exactly one cell; the host will not shrink the dispatch if you process multiple cells per thread, so extra threads just idle. Boundary cells (i==0, j==0, k==0, i==NX-1, j==NY-1, k==NZ-1) MUST copy u_curr -> u_next unchanged. Threadgroup-memory tiling and 2.5D blocking (one YX tile in shared memory, marching through Z while keeping a small Z window in registers) are the canonical optimizations for this kernel.
```

## Your previous attempt

```metal
#include <metal_stdlib>
using namespace metal;

kernel void wave_step(device const float *u_prev [[buffer(0)]],
                      device const float *u_curr [[buffer(1)]],
                      device       float *u_next [[buffer(2)]],
                      constant uint      &NX     [[buffer(3)]],
                      constant uint      &NY     [[buffer(4)]],
                      constant uint      &NZ     [[buffer(5)]],
                      constant float     &alpha  [[buffer(6)]],
                      uint3 gid [[thread_position_in_grid]]) {
    uint nx = NX;
    uint ny = NY;
    uint nz = NZ;

    uint i = gid.x;
    uint j = gid.y;
    uint k = gid.z;

    if (i >= nx || j >= ny || k >= nz) return;

    uint stride_y = nx;
    uint stride_z = nx * ny;
    uint idx = k * stride_z + j * stride_y + i;

    float c = u_curr[idx];

    // Branchless boundary evaluation using bitwise OR to avoid warp divergence
    bool is_boundary = (i == 0u) | (i == nx - 1u) | 
                       (j == 0u) | (j == ny - 1u) | 
                       (k == 0u) | (k == nz - 1u);
    if (is_boundary) {
        u_next[idx] = c;
        return;
    }

    // Explicit parallel loads to maximize memory-level parallelism
    float xm = u_curr[idx - 1u];
    float xp = u_curr[idx + 1u];
    float ym = u_curr[idx - stride_y];
    float yp = u_curr[idx + stride_y];
    float zm = u_curr[idx - stride_z];
    float zp = u_curr[idx + stride_z];
    float p  = u_prev[idx];

    // Balanced addition tree
    float sum_x = xm + xp;
    float sum_y = ym + yp;
    float sum_z = zm + zp;
    
    // Explicit Laplacian preserves spatial derivative precision
    float lap = (sum_x + sum_y + sum_z) - 6.0f * c;
    
    float a = alpha;
    
    // Contract final integration into two FMA instructions
    u_next[idx] = fma(a, lap, fma(2.0f, c, -p));
}
```

Result of previous attempt:
       64x64x64_30: correct, 2.07 ms, 45.6 GB/s (effective, 12 B/cell) (22.8% of 200 GB/s)
    160x160x160_20: correct, 6.77 ms, 145.1 GB/s (effective, 12 B/cell) (72.6% of 200 GB/s)
    192x192x192_15: correct, 8.57 ms, 148.6 GB/s (effective, 12 B/cell) (74.3% of 200 GB/s)
  score (gmean of fraction): 0.4971

## Current best (incumbent)

```metal
// Naive seed kernel for the 3D acoustic wave equation, second-order in
// time (leapfrog), 7-point spatial Laplacian.
//
//     u_next[i,j,k] = 2 u_curr[i,j,k] - u_prev[i,j,k]
//                   + alpha * ( u_curr[i-1,j,k] + u_curr[i+1,j,k]
//                             + u_curr[i,j-1,k] + u_curr[i,j+1,k]
//                             + u_curr[i,j,k-1] + u_curr[i,j,k+1]
//                             - 6 u_curr[i,j,k] )
//
// alpha = (c * dt / dx)^2; CFL stability requires alpha < 1/3 in 3D.
// Dirichlet boundary conditions: edge cells copy u_curr through unchanged.
//
// Buffer layout:
//   buffer 0: const float* u_prev   (u^{n-1}, NX * NY * NZ, row-major)
//   buffer 1: const float* u_curr   (u^n)
//   buffer 2: device float* u_next  (u^{n+1})
//   buffer 3: const uint&  NX
//   buffer 4: const uint&  NY
//   buffer 5: const uint&  NZ
//   buffer 6: const float& alpha
//
// Indexing convention: idx = (k * NY + j) * NX + i, with i the fast
// (column / x) axis, j the middle (row / y) axis, k the slow (depth / z)
// axis. The host triple-buffers across n_steps, rotating (prev, curr, next)
// each call. All dispatches share one command buffer for accurate timing.

#include <metal_stdlib>
using namespace metal;

kernel void wave_step(device const float *u_prev [[buffer(0)]],
                      device const float *u_curr [[buffer(1)]],
                      device       float *u_next [[buffer(2)]],
                      constant uint      &NX     [[buffer(3)]],
                      constant uint      &NY     [[buffer(4)]],
                      constant uint      &NZ     [[buffer(5)]],
                      constant float     &alpha  [[buffer(6)]],
                      uint3 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint j = gid.y;
    uint k = gid.z;
    if (i >= NX || j >= NY || k >= NZ) return;

    uint idx = (k * NY + j) * NX + i;

    if (i == 0u || j == 0u || k == 0u
        || i == NX - 1u || j == NY - 1u || k == NZ - 1u) {
        // Dirichlet: boundary cells stay put (u_next == u_curr).
        u_next[idx] = u_curr[idx];
        return;
    }

    uint stride_y = NX;
    uint stride_z = NX * NY;

    float c  = u_curr[idx];
    float xm = u_curr[idx - 1u];
    float xp = u_curr[idx + 1u];
    float ym = u_curr[idx - stride_y];
    float yp = u_curr[idx + stride_y];
    float zm = u_curr[idx - stride_z];
    float zp = u_curr[idx + stride_z];

    float lap = xm + xp + ym + yp + zm + zp - 6.0f * c;
    u_next[idx] = 2.0f * c - u_prev[idx] + alpha * lap;
}
```

Incumbent result:
       64x64x64_30: correct, 0.99 ms, 95.6 GB/s (effective, 12 B/cell) (47.8% of 200 GB/s)
    160x160x160_20: correct, 6.34 ms, 155.1 GB/s (effective, 12 B/cell) (77.6% of 200 GB/s)
    192x192x192_15: correct, 8.42 ms, 151.3 GB/s (effective, 12 B/cell) (75.6% of 200 GB/s)
  score (gmean of fraction): 0.6545

## History

- iter  0: compile=OK | correct=True | score=0.654462703643454
- iter  1: compile=OK | correct=True | score=0.44915440221425
- iter  2: compile=OK | correct=True | score=0.6189023155076829
- iter  3: compile=OK | correct=True | score=0.5100645157467696
- iter  4: compile=OK | correct=True | score=0.45893779639517784
- iter  5: compile=OK | correct=True | score=0.4970554266013785

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
