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

[[max_total_threads_per_threadgroup(1024)]]
kernel void wave_step(device const float *u_prev [[buffer(0)]],
                      device const float *u_curr [[buffer(1)]],
                      device       float *u_next [[buffer(2)]],
                      constant uint      &NX     [[buffer(3)]],
                      constant uint      &NY     [[buffer(4)]],
                      constant uint      &NZ     [[buffer(5)]],
                      constant float     &alpha  [[buffer(6)]],
                      uint3 gid     [[thread_position_in_grid]],
                      uint3 t_pos   [[thread_position_in_threadgroup]],
                      uint3 tg_size [[threads_per_threadgroup]]) {
    uint i = gid.x;
    uint j = gid.y;
    uint k = gid.z;
    
    uint nx = NX;
    uint ny = NY;
    uint nz = NZ;
    
    bool valid = (i < nx && j < ny && k < nz);
    uint idx = (k * ny + j) * nx + i;
    
    // Read the current cell, safely padding with 0.0 if thread is mapped outside the physical grid
    float c = valid ? u_curr[idx] : 0.0f;
    
    // Flattened thread index within the threadgroup block
    uint tid = (t_pos.z * tg_size.y + t_pos.y) * tg_size.x + t_pos.x;
    
    threadgroup float smem[1024];
    smem[tid] = c;
    
    // Ensure all threads in the group have populated their values
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Safely exit threads outside the grid only AFTER the barrier to avoid deadlocks
    if (!valid) return;

    // Dirichlet boundary condition: boundary cells copy unchanged
    bool is_boundary = (i == 0 || j == 0 || k == 0 || 
                        i == nx - 1 || j == ny - 1 || k == nz - 1);
    if (is_boundary) {
        u_next[idx] = c;
        return;
    }

    // Fetch missing neighbors from threadgroup memory if they belong to this threadgroup's block.
    // Otherwise, fallback to global memory (guaranteed to be safely in-bounds here).
    float xm = (t_pos.x > 0) ? smem[tid - 1] : u_curr[idx - 1u];
    float xp = (t_pos.x < tg_size.x - 1u) ? smem[tid + 1] : u_curr[idx + 1u];
    float lap = xm + xp;
    
    float ym = (t_pos.y > 0) ? smem[tid - tg_size.x] : u_curr[idx - nx];
    float yp = (t_pos.y < tg_size.y - 1u) ? smem[tid + tg_size.x] : u_curr[idx + nx];
    lap += ym + yp;
    
    uint stride_z = nx * ny;
    uint tg_stride_z = tg_size.x * tg_size.y;
    float zm = (t_pos.z > 0) ? smem[tid - tg_stride_z] : u_curr[idx - stride_z];
    float zp = (t_pos.z < tg_size.z - 1u) ? smem[tid + tg_stride_z] : u_curr[idx + stride_z];
    lap += zm + zp;
    
    lap -= 6.0f * c;
    
    float a = alpha;
    u_next[idx] = 2.0f * c - u_prev[idx] + a * lap;
}
```

Result of previous attempt:
       64x64x64_30: correct, 1.72 ms, 54.7 GB/s (effective, 12 B/cell) (27.4% of 200 GB/s)
    160x160x160_20: correct, 7.57 ms, 129.9 GB/s (effective, 12 B/cell) (64.9% of 200 GB/s)
    192x192x192_15: correct, 8.53 ms, 149.4 GB/s (effective, 12 B/cell) (74.7% of 200 GB/s)
  score (gmean of fraction): 0.5101

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

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
