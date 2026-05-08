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
                      uint3 gid [[thread_position_in_grid]],
                      uint3 tid [[thread_position_in_threadgroup]],
                      uint3 tpg [[threads_per_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]]) {
    const uint i = gid.x;
    const uint j = gid.y;
    const uint k = gid.z;

    if (i >= NX || j >= NY || k >= NZ) return;

    const uint stride_y = NX;
    const uint stride_z = NX * NY;
    const uint idx = k * stride_z + j * stride_y + i;

    const float c = u_curr[idx];

    const bool is_boundary =
        (i == 0u) || (j == 0u) || (k == 0u) ||
        (i == NX - 1u) || (j == NY - 1u) || (k == NZ - 1u);

    const uint lane_u = uint(lane);

    float sh_xm = c;
    float sh_xp = c;
    if (tpg.x > 1u) {
        const ushort lm = (lane == ushort(0))  ? ushort(0)  : ushort(lane - ushort(1));
        const ushort lp = (lane == ushort(31)) ? ushort(31) : ushort(lane + ushort(1));
        sh_xm = simd_shuffle(c, lm);
        sh_xp = simd_shuffle(c, lp);
    }

    float sh_ym = c;
    float sh_yp = c;
    if (tpg.y > 1u && tpg.x < 32u) {
        const uint dx = tpg.x;
        const ushort lm = (lane_u >= dx)        ? ushort(lane_u - dx) : lane;
        const ushort lp = (lane_u + dx < 32u)   ? ushort(lane_u + dx) : lane;
        sh_ym = simd_shuffle(c, lm);
        sh_yp = simd_shuffle(c, lp);
    }

    float sh_zm = c;
    float sh_zp = c;
    const uint plane_threads = tpg.x * tpg.y;
    if (tpg.z > 1u && plane_threads < 32u) {
        const uint dz = plane_threads;
        const ushort lm = (lane_u >= dz)        ? ushort(lane_u - dz) : lane;
        const ushort lp = (lane_u + dz < 32u)   ? ushort(lane_u + dz) : lane;
        sh_zm = simd_shuffle(c, lm);
        sh_zp = simd_shuffle(c, lp);
    }

    if (is_boundary) {
        u_next[idx] = c;
        return;
    }

    const bool use_xm = (tid.x != 0u) && (lane != ushort(0));
    const bool use_xp = (tid.x + 1u < tpg.x) && (lane != ushort(31));

    const bool use_ym = (tpg.y > 1u) && (tpg.x < 32u) &&
                        (tid.y != 0u) && (lane_u >= tpg.x);
    const bool use_yp = (tpg.y > 1u) && (tpg.x < 32u) &&
                        (tid.y + 1u < tpg.y) && (lane_u + tpg.x < 32u);

    const bool use_zm = (tpg.z > 1u) && (plane_threads < 32u) &&
                        (tid.z != 0u) && (lane_u >= plane_threads);
    const bool use_zp = (tpg.z > 1u) && (plane_threads < 32u) &&
                        (tid.z + 1u < tpg.z) && (lane_u + plane_threads < 32u);

    const float xm = use_xm ? sh_xm : u_curr[idx - 1u];
    const float xp = use_xp ? sh_xp : u_curr[idx + 1u];

    const float ym = use_ym ? sh_ym : u_curr[idx - stride_y];
    const float yp = use_yp ? sh_yp : u_curr[idx + stride_y];

    const float zm = use_zm ? sh_zm : u_curr[idx - stride_z];
    const float zp = use_zp ? sh_zp : u_curr[idx + stride_z];

    const float sum = (xm + xp) + (ym + yp) + (zm + zp);
    const float lap = sum - 6.0f * c;

    u_next[idx] = fma(alpha, lap, (c + c) - u_prev[idx]);
}
```

Result of previous attempt:
       64x64x64_30: correct, 2.31 ms, 40.8 GB/s (effective, 12 B/cell) (20.4% of 200 GB/s)
    160x160x160_20: correct, 12.05 ms, 81.6 GB/s (effective, 12 B/cell) (40.8% of 200 GB/s)
    192x192x192_15: correct, 15.68 ms, 81.2 GB/s (effective, 12 B/cell) (40.6% of 200 GB/s)
  score (gmean of fraction): 0.3234

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
       64x64x64_30: correct, 0.79 ms, 118.8 GB/s (effective, 12 B/cell) (59.4% of 200 GB/s)
    160x160x160_20: correct, 6.89 ms, 142.6 GB/s (effective, 12 B/cell) (71.3% of 200 GB/s)
    192x192x192_15: correct, 8.56 ms, 148.9 GB/s (effective, 12 B/cell) (74.4% of 200 GB/s)
  score (gmean of fraction): 0.6808

## History

- iter  0: compile=OK | correct=True | score=0.6807613316597116
- iter  1: compile=OK | correct=True | score=0.3234463742438968

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
