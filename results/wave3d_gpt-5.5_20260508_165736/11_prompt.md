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

kernel void wave_step(device const float *__restrict__ u_prev [[buffer(0)]],
                      device const float *__restrict__ u_curr [[buffer(1)]],
                      device       float *__restrict__ u_next [[buffer(2)]],
                      constant uint      &NX     [[buffer(3)]],
                      constant uint      &NY     [[buffer(4)]],
                      constant uint      &NZ     [[buffer(5)]],
                      constant float     &alpha  [[buffer(6)]],
                      uint3 gid  [[thread_position_in_grid]],
                      uint3 ltid [[thread_position_in_threadgroup]],
                      uint3 tgdim [[threads_per_threadgroup]],
                      uint  tidx [[thread_index_in_threadgroup]],
                      uint  lane [[thread_index_in_simdgroup]])
{
    threadgroup float tile[1024];

    const uint nx = NX;
    const uint ny = NY;
    const uint nz = NZ;

    const uint i = gid.x;
    const uint j = gid.y;
    const uint k = gid.z;

    const uint stride_y = nx;
    const uint stride_z = nx * ny;
    const uint idx = (k * ny + j) * nx + i;

    const uint tg_count = tgdim.x * tgdim.y * tgdim.z;

    // For the 64^3 case the hardware caches are usually enough; avoid
    // barriers/shared-memory overhead.  For large groups with real Z depth,
    // use threadgroup tiling.  Otherwise use a barrier-free SIMD shuffle path.
    const bool small_grid = (nx <= 96u) | (ny <= 96u) | (nz <= 96u);
    const bool use_tgm =
        (!small_grid) &
        (tg_count >= 128u) & (tg_count <= 1024u) &
        (tgdim.x >= 4u) & (tgdim.y >= 2u) & (tgdim.z >= 2u);

    if (small_grid) {
        if ((i >= nx) | (j >= ny) | (k >= nz)) return;

        const float c = u_curr[idx];

        const bool interior =
            (i != 0u) & (j != 0u) & (k != 0u) &
            (i < (nx - 1u)) & (j < (ny - 1u)) & (k < (nz - 1u));

        if (!interior) {
            u_next[idx] = c;
            return;
        }

        const float xm = u_curr[idx - 1u];
        const float xp = u_curr[idx + 1u];
        const float ym = u_curr[idx - stride_y];
        const float yp = u_curr[idx + stride_y];
        const float zm = u_curr[idx - stride_z];
        const float zp = u_curr[idx + stride_z];

        const float lap = xm + xp + ym + yp + zm + zp - 6.0f * c;
        u_next[idx] = 2.0f * c - u_prev[idx] + alpha * lap;
        return;
    }

    if (!use_tgm) {
        const bool in_bounds = (i < nx) & (j < ny) & (k < nz);

        float c = 0.0f;
        if (in_bounds) {
            c = u_curr[idx];
        }

        const uint lane_u = lane;

        bool can_xm = false;
        bool can_xp = false;
        bool can_ym = false;
        bool can_yp = false;
        bool can_zm = false;
        bool can_zp = false;

        float sh_xm = 0.0f;
        float sh_xp = 0.0f;
        float sh_ym = 0.0f;
        float sh_yp = 0.0f;
        float sh_zm = 0.0f;
        float sh_zp = 0.0f;

        if (tgdim.x > 1u) {
            can_xm = (ltid.x != 0u) & (lane_u != 0u);
            can_xp = ((ltid.x + 1u) < tgdim.x) & ((lane_u + 1u) < 32u);

            const uint xm_lane = can_xm ? (lane_u - 1u) : lane_u;
            const uint xp_lane = can_xp ? (lane_u + 1u) : lane_u;

            sh_xm = simd_shuffle(c, ushort(xm_lane));
            sh_xp = simd_shuffle(c, ushort(xp_lane));
        }

        if ((tgdim.x < 32u) & (tgdim.y > 1u)) {
            const uint sx = tgdim.x;

            can_ym = (ltid.y != 0u) & (lane_u >= sx);
            can_yp = ((ltid.y + 1u) < tgdim.y) & ((lane_u + sx) < 32u);

            const uint ym_lane = can_ym ? (lane_u - sx) : lane_u;
            const uint yp_lane = can_yp ? (lane_u + sx) : lane_u;

            sh_ym = simd_shuffle(c, ushort(ym_lane));
            sh_yp = simd_shuffle(c, ushort(yp_lane));
        }

        const uint sxy = tgdim.x * tgdim.y;
        if ((sxy < 32u) & (tgdim.z > 1u)) {
            can_zm = (ltid.z != 0u) & (lane_u >= sxy);
            can_zp = ((ltid.z + 1u) < tgdim.z) & ((lane_u + sxy) < 32u);

            const uint zm_lane = can_zm ? (lane_u - sxy) : lane_u;
            const uint zp_lane = can_zp ? (lane_u + sxy) : lane_u;

            sh_zm = simd_shuffle(c, ushort(zm_lane));
            sh_zp = simd_shuffle(c, ushort(zp_lane));
        }

        if (!in_bounds) return;

        const bool interior =
            (i != 0u) & (j != 0u) & (k != 0u) &
            (i < (nx - 1u)) & (j < (ny - 1u)) & (k < (nz - 1u));

        if (!interior) {
            u_next[idx] = c;
            return;
        }

        float xm = sh_xm;
        float xp = sh_xp;
        float ym = sh_ym;
        float yp = sh_yp;
        float zm = sh_zm;
        float zp = sh_zp;

        if (!can_xm) xm = u_curr[idx - 1u];
        if (!can_xp) xp = u_curr[idx + 1u];
        if (!can_ym) ym = u_curr[idx - stride_y];
        if (!can_yp) yp = u_curr[idx + stride_y];
        if (!can_zm) zm = u_curr[idx - stride_z];
        if (!can_zp) zp = u_curr[idx + stride_z];

        const float lap = xm + xp + ym + yp + zm + zp - 6.0f * c;
        u_next[idx] = 2.0f * c - u_prev[idx] + alpha * lap;
        return;
    }

    // Large-grid path: stage the current threadgroup's 3D block of u_curr.
    const bool in_bounds = (i < nx) & (j < ny) & (k < nz);

    float c = 0.0f;
    if (in_bounds) {
        c = u_curr[idx];
    }

    tile[tidx] = c;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!in_bounds) return;

    const bool interior =
        (i != 0u) & (j != 0u) & (k != 0u) &
        (i < (nx - 1u)) & (j < (ny - 1u)) & (k < (nz - 1u));

    if (!interior) {
        u_next[idx] = c;
        return;
    }

    const uint sx  = tgdim.x;
    const uint sxy2 = tgdim.x * tgdim.y;

    const float xm = (ltid.x != 0u)            ? tile[tidx - 1u]    : u_curr[idx - 1u];
    const float xp = ((ltid.x + 1u) < tgdim.x) ? tile[tidx + 1u]    : u_curr[idx + 1u];

    const float ym = (ltid.y != 0u)            ? tile[tidx - sx]    : u_curr[idx - stride_y];
    const float yp = ((ltid.y + 1u) < tgdim.y) ? tile[tidx + sx]    : u_curr[idx + stride_y];

    const float zm = (ltid.z != 0u)            ? tile[tidx - sxy2]  : u_curr[idx - stride_z];
    const float zp = ((ltid.z + 1u) < tgdim.z) ? tile[tidx + sxy2]  : u_curr[idx + stride_z];

    const float lap = xm + xp + ym + yp + zm + zp - 6.0f * c;
    u_next[idx] = 2.0f * c - u_prev[idx] + alpha * lap;
}
```

Result of previous attempt:
       64x64x64_30: correct, 2.17 ms, 43.5 GB/s (effective, 12 B/cell) (21.7% of 200 GB/s)
    160x160x160_20: correct, 10.27 ms, 95.7 GB/s (effective, 12 B/cell) (47.9% of 200 GB/s)
    192x192x192_15: correct, 13.40 ms, 95.1 GB/s (effective, 12 B/cell) (47.5% of 200 GB/s)
  score (gmean of fraction): 0.3671

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

- iter  3: compile=OK | correct=True | score=0.42638373245244165
- iter  4: compile=OK | correct=True | score=0.48940570156473123
- iter  5: compile=OK | correct=True | score=0.275344915512935
- iter  6: compile=OK | correct=True | score=0.5212906034041827
- iter  7: compile=OK | correct=True | score=0.5252265482130357
- iter  8: compile=OK | correct=True | score=0.48114767809612935
- iter  9: compile=FAIL | correct=False | score=N/A
- iter 10: compile=OK | correct=True | score=0.36710750656074076

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
