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

kernel void wave_step(device const float * restrict u_prev [[buffer(0)]],
                      device const float * restrict u_curr [[buffer(1)]],
                      device       float * restrict u_next [[buffer(2)]],
                      constant uint      &NX     [[buffer(3)]],
                      constant uint      &NY     [[buffer(4)]],
                      constant uint      &NZ     [[buffer(5)]],
                      constant float     &alpha  [[buffer(6)]],
                      uint3 gid [[thread_position_in_grid]],
                      uint3 tpos [[thread_position_in_threadgroup]],
                      uint3 tsize [[threads_per_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint simdW [[thread_execution_width]]) {
    
    uint nx = NX;
    uint ny = NY;
    uint nz = NZ;
    
    uint i = gid.x;
    uint j = gid.y;
    uint k = gid.z;
    
    // Bounds check
    if (i >= nx || j >= ny || k >= nz) return;

    uint idx = (k * ny + j) * nx + i;
    
    // Every active thread loads its center element
    float c = u_curr[idx];
    
    uint tx = tpos.x;
    
    // SIMD group cooperation for X-axis neighbors (bypasses memory loads)
    float xm = simd_shuffle_up(c, 1);
    // Overwrite with memory load if this thread is the first in its row or SIMD group
    if ((tx == 0 || lane == 0) && i > 0) {
        xm = u_curr[idx - 1];
    }
    
    float xp = simd_shuffle_down(c, 1);
    // Overwrite with memory load if this thread is the last in its row or SIMD group
    if ((tx == tsize.x - 1 || lane == simdW - 1) && i < nx - 1) {
        xp = u_curr[idx + 1];
    }
    
    // Dirichlet boundary condition
    // We evaluate this AFTER the shuffle so boundary cells can supply 'c' to interior cells
    bool is_boundary = (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1);
    if (is_boundary) {
        u_next[idx] = c;
        return;
    }
    
    // Only interior cells proceed to load Y and Z neighbors and compute the update
    uint stride_y = nx;
    uint stride_z = nx * ny;
    
    float p = u_prev[idx];
    
    float ym = u_curr[idx - stride_y];
    float yp = u_curr[idx + stride_y];
    float zm = u_curr[idx - stride_z];
    float zp = u_curr[idx + stride_z];
    
    float lap = (xm + xp) + (ym + yp) + (zm + zp) - 6.0f * c;
    float a = alpha;
    
    u_next[idx] = fma(a, lap, fma(2.0f, c, -p));
}
```

Result of previous attempt:
  COMPILE FAILED: Error Domain=MTLLibraryErrorDomain Code=3 "program_source:4:53: error: expected ')'
kernel void wave_step(device const float * restrict u_prev [[buffer(0)]],
                                                    ^
program_source:4:22: note: to match this '('
kernel void wave_step(device const float * restrict u_prev [[buffer(0)]],
                     ^
program_source:17:15: error: use of undeclared identifier 'NX'
    uint nx = NX;
              ^
program_source:18:15: error: use of undeclared identifier 'NY'
    uint ny = NY;
              ^
program_source:19:15: error: use of undeclared identifier 'NZ'
    uint nz = NZ;
              ^
program_source:21:14: error: use of undeclared identifier 'gid'
    uint i = gid.x;
             ^
program_source:22:14: error: use of undeclared identifier 'gid'
    uint j = gid.y;
             ^
program_source:23:14: error: use of undeclared identifier 'gid'
    uint k = gid.z;
             ^
program_source:31:15: error: use of undeclared identifier 'u_curr'
    float c = u_curr[idx];
              ^
program_source:33:15: error: use of undeclared identifier 'tpos'
    uint tx = tpos.x;
              ^
program_source:38:21: error: use of undeclared identifier 'lane'
    if ((tx == 0 || lane == 0) && i > 0) {
                    ^
program_source:39:14: error: use of undeclared identifier 'u_curr'
        xm = u_curr[idx - 1];
             ^
program_source:44:16: error: use of undeclared identifier 'tsize'
    if ((tx == tsize.x - 1 || lane == simdW - 1) && i < nx - 1) {
               ^
program_source:44:31: error: use of undeclared identifier 'lane'
    if ((tx == tsize.x - 1 || lane == simdW - 1) && i < nx - 1) {
                              ^
program_source:44:39: error: use of undeclared identifier 'simdW'
    if ((tx == tsize.x - 1 || lane == simdW - 1) && i < nx - 1) {
                                      ^
program_source:45:14: error: use of undeclared identifier 'u_curr'
        xp = u_curr[idx + 1];
             ^
program_source:52:9: error: use of undeclared identifier 'u_next'
        u_next[idx] = c;
        ^
program_source:60:15: error: use of undeclared identifier 'u_prev'
    float p = u_prev[idx];
              ^
program_source:62:16: error: use of undeclared identifier 'u_curr'
    float ym = u_curr[idx - stride_y];
               ^
program_source:63:16: error: use of undeclared identifier 'u_curr'
    float yp = u_curr[idx + stride_y];
               ^
program_source:64:16: error: use of undeclared identifier 'u_curr'
    float zm = u_curr[idx - stride_z];
               ^
program_source:65:16: error: use of undeclared identifier 'u_curr'
    float zp = u_curr[idx + stride_z];
               ^
program_source:68:15: error: use of undeclared identifier 'alpha'
    float a = alpha;
              ^
program_source:70:5: error: use of undeclared identifier 'u_next'
    u_next[idx] = fma(a, lap, fma(2.0f, c, -p));
    ^
" UserInfo={NSLocalizedDescription=program_source:4:53: error: expected ')'
kernel void wave_step(device const float * restrict u_prev [[buffer(0)]],
                                                    ^
program_source:4:22: note: to match this '('
kernel void wave_step(device const float * restrict u_prev [[buffer(0)]],
                     ^
program_source:17:15: error: use of undeclared identifier 'NX'
    uint nx = NX;
              ^
program_source:18:15: error: use of undeclared identifier 'NY'
    uint ny = NY;
              ^
program_source:19:15: error: use of undeclared identifier 'NZ'
    uint nz = NZ;
              ^
program_source:21:14: error: use of undeclared identifier 'gid'
    uint i = gid.x;
             ^
program_source:22:14: error: use of undeclared identifier 'gid'
    uint j = gid.y;
             ^
program_source:23:14: error: use of undeclared identifier 'gid'
    uint k = gid.z;
             ^
program_source:31:15: error: use of undeclared identifier 'u_curr'
    float c = u_curr[idx];
              ^
program_source:33:15: error: use of undeclared identifier 'tpos'
    uint tx = tpos.x;
              ^
program_source:38:21: error: use of undeclared identifier 'lane'
    if ((tx == 0 || lane == 0) && i > 0) {
                    ^
program_source:39:14: error: use of undeclared identifier 'u_curr'
        xm = u_curr[idx - 1];
             ^
program_source:44:16: error: use of undeclared identifier 'tsize'
    if ((tx == tsize.x - 1 || lane == simdW - 1) && i < nx - 1) {
               ^
program_source:44:31: error: use of undeclared identifier 'lane'
    if ((tx == tsize.x - 1 || lane == simdW - 1) && i < nx - 1) {
                              ^
program_source:44:39: error: use of undeclared identifier 'simdW'
    if ((tx == tsize.x - 1 || lane == simdW - 1) && i < nx - 1) {
                                      ^
program_source:45:14: error: use of undeclared identifier 'u_curr'
        xp = u_curr[idx + 1];
             ^
program_source:52:9: error: use of undeclared identifier 'u_next'
        u_next[idx] = c;
        ^
program_source:60:15: error: use of undeclared identifier 'u_prev'
    float p = u_prev[idx];
              ^
program_source:62:16: error: use of undeclared identifier 'u_curr'
    float ym = u_curr[idx - stride_y];
               ^
program_source:63:16: error: use of undeclared identifier 'u_curr'
    float yp = u_curr[idx + stride_y];
               ^
program_source:64:16: error: use of undeclared identifier 'u_curr'
    float zm = u_curr[idx - stride_z];
               ^
program_source:65:16: error: use of undeclared identifier 'u_curr'
    float zp = u_curr[idx + stride_z];
               ^
program_source:68:15: error: use of undeclared identifier 'alpha'
    float a = alpha;
              ^
program_source:70:5: error: use of undeclared identifier 'u_next'
    u_next[idx] = fma(a, lap, fma(2.0f, c, -p));
    ^
}

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
- iter  6: compile=FAIL | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
