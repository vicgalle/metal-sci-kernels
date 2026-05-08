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
                      uint2 tpg [[threads_per_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    const uint i = gid.x;
    const uint j = gid.y;
    const uint nx = NX;
    const uint ny = NY;

    if (i >= nx || j >= ny) return;

    const uint idx = j * nx + i;

    // For small grids the launch/branch overhead dominates; keep the compact
    // scalar path that compiles very well.
    if (nx <= 256u) {
        if (i == 0u || j == 0u || i == nx - 1u || j == ny - 1u) {
            u_out[idx] = u_in[idx];
            return;
        }

        const float c = u_in[idx];
        const float l = u_in[idx - 1u];
        const float r = u_in[idx + 1u];
        const float d = u_in[idx - nx];
        const float u = u_in[idx + nx];
        u_out[idx] = c + alpha * (l + r + d + u - 4.0f * c);
        return;
    }

    // Load center for every in-grid lane, including boundary lanes, before any
    // early return so neighboring interior lanes may reuse it via SIMD shuffle.
    const float c = u_in[idx];

    const uint left_lane  = (lane == 0u)  ? lane : (lane - 1u);
    const uint right_lane = (lane == 31u) ? lane : (lane + 1u);

    const float sh_l = simd_shuffle(c, ushort(left_lane));
    const float sh_r = simd_shuffle(c, ushort(right_lane));

    float sh_d = c;
    float sh_u = c;
    bool use_sh_d = false;
    bool use_sh_u = false;

    // If a row is narrower than the SIMD width, some vertical neighbors are
    // also resident in the same SIMDgroup at lane +/- tpg.x.
    if (tpg.x < 32u) {
        const uint up_lane = (lane >= tpg.x) ? (lane - tpg.x) : lane;
        const uint down_lane_raw = lane + tpg.x;
        const uint down_lane = (down_lane_raw < 32u) ? down_lane_raw : lane;

        sh_d = simd_shuffle(c, ushort(up_lane));
        sh_u = simd_shuffle(c, ushort(down_lane));

        use_sh_d = (tid.y != 0u) && (lane >= tpg.x);
        use_sh_u = ((tid.y + 1u) < tpg.y) && (down_lane_raw < 32u);
    }

    if (i == 0u || j == 0u || i == nx - 1u || j == ny - 1u) {
        u_out[idx] = c;
        return;
    }

    const bool use_sh_l = (lane != 0u)  && (tid.x != 0u);
    const bool use_sh_r = (lane != 31u) && ((tid.x + 1u) < tpg.x);

    const float l = use_sh_l ? sh_l : u_in[idx - 1u];
    const float r = use_sh_r ? sh_r : u_in[idx + 1u];
    const float d = use_sh_d ? sh_d : u_in[idx - nx];
    const float u = use_sh_u ? sh_u : u_in[idx + nx];

    u_out[idx] = c + alpha * (l + r + d + u - 4.0f * c);
}
```

Result of previous attempt:
        256x256_50: correct, 0.72 ms, 36.2 GB/s (effective, 8 B/cell) (18.1% of 200 GB/s)
       512x512_100: correct, 2.97 ms, 70.7 GB/s (effective, 8 B/cell) (35.4% of 200 GB/s)
      1024x1024_50: correct, 5.17 ms, 81.2 GB/s (effective, 8 B/cell) (40.6% of 200 GB/s)
  score (gmean of fraction): 0.2961

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
        256x256_50: correct, 0.37 ms, 70.1 GB/s (effective, 8 B/cell) (35.1% of 200 GB/s)
       512x512_100: correct, 1.26 ms, 165.8 GB/s (effective, 8 B/cell) (82.9% of 200 GB/s)
      1024x1024_50: correct, 1.87 ms, 224.3 GB/s (effective, 8 B/cell) (112.2% of 200 GB/s)
  score (gmean of fraction): 0.6883

## History

- iter  0: compile=OK | correct=True | score=0.6883214595661554
- iter  1: compile=OK | correct=True | score=0.36474589028668897
- iter  2: compile=OK | correct=True | score=0.3463215725221932
- iter  3: compile=OK | correct=True | score=0.2961192957141125

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
