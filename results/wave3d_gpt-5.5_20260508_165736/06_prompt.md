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
                      uint lane [[thread_index_in_simdgroup]],
                      uint3 tpg [[threads_per_threadgroup]]) {
    const uint i = gid.x;
    const uint j = gid.y;
    const uint k = gid.z;

    const uint nx = NX;
    const uint ny = NY;
    const uint nz = NZ;

    const bool in_bounds = (i < nx) && (j < ny) && (k < nz);

    const uint stride_y = nx;
    const uint stride_z = nx * ny;
    const uint idx = (k * ny + j) * nx + i;

    float c = 0.0f;
    if (in_bounds) {
        c = u_curr[idx];
    }

    // Keep all lanes participating in the SIMD exchange, including boundary
    // lanes and any padded out-of-grid lanes.  Invalid lanes publish a sentinel
    // tag so they can never be accepted as a stencil neighbor.
    const uint invalid_tag = 0xffffffffu;
    const uint tag = in_bounds ? idx : invalid_tag;

    const ushort self = (ushort)lane;

    const ushort src_xm = (lane > 0u)  ? (ushort)(lane - 1u) : self;
    const ushort src_xp = (lane < 31u) ? (ushort)(lane + 1u) : self;

    const uint  tag_xm = simd_shuffle(tag, src_xm);
    const uint  tag_xp = simd_shuffle(tag, src_xp);
    const float val_xm = simd_shuffle(c,   src_xm);
    const float val_xp = simd_shuffle(c,   src_xp);

    uint  tag_ym = invalid_tag;
    uint  tag_yp = invalid_tag;
    float val_ym = 0.0f;
    float val_yp = 0.0f;

    const uint dy = tpg.x;
    if (dy < 32u) {
        const ushort src_ym = (lane >= dy)        ? (ushort)(lane - dy) : self;
        const ushort src_yp = (lane + dy < 32u)   ? (ushort)(lane + dy) : self;

        tag_ym = simd_shuffle(tag, src_ym);
        tag_yp = simd_shuffle(tag, src_yp);
        val_ym = simd_shuffle(c,   src_ym);
        val_yp = simd_shuffle(c,   src_yp);
    }

    uint  tag_zm = invalid_tag;
    uint  tag_zp = invalid_tag;
    float val_zm = 0.0f;
    float val_zp = 0.0f;

    const uint dz = tpg.x * tpg.y;
    if (dz < 32u) {
        const ushort src_zm = (lane >= dz)        ? (ushort)(lane - dz) : self;
        const ushort src_zp = (lane + dz < 32u)   ? (ushort)(lane + dz) : self;

        tag_zm = simd_shuffle(tag, src_zm);
        tag_zp = simd_shuffle(tag, src_zp);
        val_zm = simd_shuffle(c,   src_zm);
        val_zp = simd_shuffle(c,   src_zp);
    }

    if (!in_bounds) {
        return;
    }

    if (i == 0u || j == 0u || k == 0u ||
        i == nx - 1u || j == ny - 1u || k == nz - 1u) {
        u_next[idx] = c;
        return;
    }

    float xm;
    if (tag_xm == idx - 1u) {
        xm = val_xm;
    } else {
        xm = u_curr[idx - 1u];
    }

    float xp;
    if (tag_xp == idx + 1u) {
        xp = val_xp;
    } else {
        xp = u_curr[idx + 1u];
    }

    float ym;
    if (tag_ym == idx - stride_y) {
        ym = val_ym;
    } else {
        ym = u_curr[idx - stride_y];
    }

    float yp;
    if (tag_yp == idx + stride_y) {
        yp = val_yp;
    } else {
        yp = u_curr[idx + stride_y];
    }

    float zm;
    if (tag_zm == idx - stride_z) {
        zm = val_zm;
    } else {
        zm = u_curr[idx - stride_z];
    }

    float zp;
    if (tag_zp == idx + stride_z) {
        zp = val_zp;
    } else {
        zp = u_curr[idx + stride_z];
    }

    const float lap = xm + xp + ym + yp + zm + zp - 6.0f * c;
    u_next[idx] = 2.0f * c - u_prev[idx] + alpha * lap;
}
```

Result of previous attempt:
       64x64x64_30: correct, 2.15 ms, 43.8 GB/s (effective, 12 B/cell) (21.9% of 200 GB/s)
    160x160x160_20: correct, 15.93 ms, 61.7 GB/s (effective, 12 B/cell) (30.8% of 200 GB/s)
    192x192x192_15: correct, 20.63 ms, 61.8 GB/s (effective, 12 B/cell) (30.9% of 200 GB/s)
  score (gmean of fraction): 0.2753

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
- iter  2: compile=OK | correct=True | score=0.4886586399087577
- iter  3: compile=OK | correct=True | score=0.42638373245244165
- iter  4: compile=OK | correct=True | score=0.48940570156473123
- iter  5: compile=OK | correct=True | score=0.275344915512935

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
