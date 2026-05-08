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

#define BX 32
#define BY 8
#define BZ 4

kernel void wave_step(device const float *u_prev [[buffer(0)]],
                      device const float *u_curr [[buffer(1)]],
                      device       float *u_next [[buffer(2)]],
                      constant uint      &NX     [[buffer(3)]],
                      constant uint      &NY     [[buffer(4)]],
                      constant uint      &NZ     [[buffer(5)]],
                      constant float     &alpha  [[buffer(6)]],
                      uint3 gid [[thread_position_in_grid]],
                      uint3 lid [[thread_position_in_threadgroup]]) {
    threadgroup float tile[BZ][BY+2][BX+2];

    uint i = gid.x;
    uint j = gid.y;
    uint k = gid.z;

    uint stride_y = NX;
    uint stride_z = NX * NY;

    uint lx = lid.x + 1u;
    uint ly = lid.y + 1u;
    uint lz = lid.z;

    bool in_bounds = (i < NX) && (j < NY) && (k < NZ);

    // Center load
    if (in_bounds) {
        tile[lz][ly][lx] = u_curr[k * stride_z + j * stride_y + i];
    } else {
        tile[lz][ly][lx] = 0.0f;
    }

    // X halos
    if (lid.x == 0u) {
        uint ii = (i == 0u) ? 0u : i - 1u;
        if (j < NY && k < NZ) {
            tile[lz][ly][0] = u_curr[k * stride_z + j * stride_y + ii];
        } else {
            tile[lz][ly][0] = 0.0f;
        }
    }
    if (lid.x == BX - 1u || i == NX - 1u) {
        uint ii = (i + 1u >= NX) ? NX - 1u : i + 1u;
        if (j < NY && k < NZ) {
            tile[lz][ly][lx + 1u] = u_curr[k * stride_z + j * stride_y + ii];
        } else {
            tile[lz][ly][lx + 1u] = 0.0f;
        }
    }
    // Y halos
    if (lid.y == 0u) {
        uint jj = (j == 0u) ? 0u : j - 1u;
        if (i < NX && k < NZ) {
            tile[lz][0][lx] = u_curr[k * stride_z + jj * stride_y + i];
        } else {
            tile[lz][0][lx] = 0.0f;
        }
    }
    if (lid.y == BY - 1u || j == NY - 1u) {
        uint jj = (j + 1u >= NY) ? NY - 1u : j + 1u;
        if (i < NX && k < NZ) {
            tile[lz][ly + 1u][lx] = u_curr[k * stride_z + jj * stride_y + i];
        } else {
            tile[lz][ly + 1u][lx] = 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!in_bounds) return;

    uint idx = k * stride_z + j * stride_y + i;

    if (i == 0u || j == 0u || k == 0u
        || i == NX - 1u || j == NY - 1u || k == NZ - 1u) {
        u_next[idx] = u_curr[idx];
        return;
    }

    float c  = tile[lz][ly][lx];
    float xm = tile[lz][ly][lx - 1u];
    float xp = tile[lz][ly][lx + 1u];
    float ym = tile[lz][ly - 1u][lx];
    float yp = tile[lz][ly + 1u][lx];
    float zm = u_curr[idx - stride_z];
    float zp = u_curr[idx + stride_z];

    float lap = xm + xp + ym + yp + zm + zp - 6.0f * c;
    u_next[idx] = 2.0f * c - u_prev[idx] + alpha * lap;
}
```

Result of previous attempt:
       64x64x64_30: INCORRECT (max_abs=8.338e-01, tol=0.00010400429666042328)
  fail_reason: correctness failed at size 64x64x64_30: max_abs=8.338e-01

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
       64x64x64_30: correct, 1.53 ms, 61.9 GB/s (effective, 12 B/cell) (30.9% of 200 GB/s)
    160x160x160_20: correct, 6.31 ms, 155.8 GB/s (effective, 12 B/cell) (77.9% of 200 GB/s)
    192x192x192_15: correct, 8.29 ms, 153.7 GB/s (effective, 12 B/cell) (76.9% of 200 GB/s)
  score (gmean of fraction): 0.5700

## History

- iter  0: compile=OK | correct=True | score=0.5699974674134811
- iter  1: compile=OK | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
