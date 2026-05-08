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
    const uint i = gid.x;
    const uint j = gid.y;
    const uint k = gid.z;
    if (i >= NX || j >= NY || k >= NZ) return;

    const uint NXm1 = NX - 1u;
    const uint NYm1 = NY - 1u;
    const uint NZm1 = NZ - 1u;

    const uint stride_y = NX;
    const uint stride_z = NX * NY;
    const uint idx = (k * NY + j) * NX + i;

    if (i == 0u || j == 0u || k == 0u
        || i == NXm1 || j == NYm1 || k == NZm1) {
        u_next[idx] = u_curr[idx];
        return;
    }

    float c  = u_curr[idx];
    float xm = u_curr[idx - 1u];
    float xp = u_curr[idx + 1u];
    float ym = u_curr[idx - stride_y];
    float yp = u_curr[idx + stride_y];
    float zm = u_curr[idx - stride_z];
    float zp = u_curr[idx + stride_z];
    float p  = u_prev[idx];

    float sxy = (xm + xp) + (ym + yp);
    float sz  = zm + zp;
    float lap = (sxy + sz) - 6.0f * c;

    u_next[idx] = fma(alpha, lap, fma(2.0f, c, -p));
}
```

Result of previous attempt:
       64x64x64_30: correct, 0.85 ms, 110.6 GB/s (effective, 12 B/cell) (55.3% of 200 GB/s)
    160x160x160_20: correct, 6.87 ms, 143.0 GB/s (effective, 12 B/cell) (71.5% of 200 GB/s)
    192x192x192_15: correct, 8.26 ms, 154.2 GB/s (effective, 12 B/cell) (77.1% of 200 GB/s)
  score (gmean of fraction): 0.6731

## History

- iter  7: compile=OK | correct=False | score=N/A
- iter  8: compile=OK | correct=False | score=N/A
- iter  9: compile=FAIL | correct=False | score=N/A
- iter 10: compile=OK | correct=False | score=N/A
- iter 11: compile=OK | correct=False | score=N/A
- iter 12: compile=OK | correct=False | score=N/A
- iter 13: compile=OK | correct=False | score=N/A
- iter 14: compile=OK | correct=True | score=0.6730763254297923

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
