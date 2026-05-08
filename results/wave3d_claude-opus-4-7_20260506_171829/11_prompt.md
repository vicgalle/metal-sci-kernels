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

#define TX 32u
#define TY 8u
#define SX (TX + 2u)
#define SY (TY + 2u)

kernel void wave_step(device const float *u_prev [[buffer(0)]],
                      device const float *u_curr [[buffer(1)]],
                      device       float *u_next [[buffer(2)]],
                      constant uint      &NX     [[buffer(3)]],
                      constant uint      &NY     [[buffer(4)]],
                      constant uint      &NZ     [[buffer(5)]],
                      constant float     &alpha  [[buffer(6)]],
                      uint3 gid  [[thread_position_in_grid]],
                      uint3 tid  [[thread_position_in_threadgroup]],
                      uint3 tgid [[threadgroup_position_in_grid]]) {
    threadgroup float tile[SY][SX];

    const uint i = gid.x;
    const uint j = gid.y;
    const uint k = gid.z;

    const uint lx = tid.x;
    const uint ly = tid.y;

    const uint NXm1 = NX - 1u;
    const uint NYm1 = NY - 1u;
    const uint NZm1 = NZ - 1u;

    const uint stride_z = NX * NY;

    const uint i0 = tgid.x * TX;
    const uint j0 = tgid.y * TY;

    const uint kc = min(k, NZm1);
    const uint lid = ly * TX + lx;
    const uint total = TX * TY;
    const uint tile_n = SX * SY;

    // Cooperative halo load of (TY+2) x (TX+2) tile from u_curr at slice kc.
    for (uint t = lid; t < tile_n; t += total) {
        uint sxi = t % SX;
        uint syi = t / SX;
        int gi = (int)i0 + (int)sxi - 1;
        int gj = (int)j0 + (int)syi - 1;
        gi = clamp(gi, 0, (int)NXm1);
        gj = clamp(gj, 0, (int)NYm1);
        tile[syi][sxi] = u_curr[(kc * NY + (uint)gj) * NX + (uint)gi];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (i >= NX || j >= NY || k >= NZ) return;

    const uint idx = (k * NY + j) * NX + i;

    if (i == 0u || j == 0u || k == 0u
        || i == NXm1 || j == NYm1 || k == NZm1) {
        u_next[idx] = u_curr[idx];
        return;
    }

    const uint sx = lx + 1u;
    const uint sy = ly + 1u;

    float c  = tile[sy][sx];
    float xm = tile[sy][sx - 1u];
    float xp = tile[sy][sx + 1u];
    float ym = tile[sy - 1u][sx];
    float yp = tile[sy + 1u][sx];
    float zm = u_curr[idx - stride_z];
    float zp = u_curr[idx + stride_z];

    float lap = (xm + xp) + (ym + yp) + (zm + zp) - 6.0f * c;
    float p   = u_prev[idx];
    u_next[idx] = fma(alpha, lap, fma(2.0f, c, -p));
}
```

Result of previous attempt:
       64x64x64_30: INCORRECT (max_abs=nan, tol=0.00010400429666042328)
  fail_reason: correctness failed at size 64x64x64_30: max_abs=nan

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
       64x64x64_30: correct, 1.87 ms, 50.4 GB/s (effective, 12 B/cell) (25.2% of 200 GB/s)
    160x160x160_20: correct, 6.29 ms, 156.2 GB/s (effective, 12 B/cell) (78.1% of 200 GB/s)
    192x192x192_15: correct, 8.30 ms, 153.5 GB/s (effective, 12 B/cell) (76.8% of 200 GB/s)
  score (gmean of fraction): 0.5325

## History

- iter  3: compile=OK | correct=False | score=N/A
- iter  4: compile=OK | correct=True | score=0.5082674390658176
- iter  5: compile=OK | correct=True | score=0.5031775738333382
- iter  6: compile=OK | correct=False | score=N/A
- iter  7: compile=OK | correct=False | score=N/A
- iter  8: compile=OK | correct=False | score=N/A
- iter  9: compile=FAIL | correct=False | score=N/A
- iter 10: compile=OK | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
