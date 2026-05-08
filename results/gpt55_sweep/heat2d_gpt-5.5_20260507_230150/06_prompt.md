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
                      uint2 lid [[thread_position_in_threadgroup]],
                      uint2 tpg [[threads_per_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    const uint i  = gid.x;
    const uint j  = gid.y;
    const uint nx = NX;
    const uint ny = NY;

    if (i >= nx || j >= ny) return;

    const uint idx = j * nx + i;
    const float c = u_in[idx];

    // All valid lanes, including boundary lanes, participate so neighboring
    // interior lanes can read boundary values through the SIMD shuffle.
    const float l_sh = simd_shuffle_up(c, (ushort)1);
    const float r_sh = simd_shuffle_down(c, (ushort)1);

    const uint tx = tpg.x;
    const bool do_v_shuffle = (tx <= 16u) && (tpg.y > 1u);

    float u_sh = c;
    float d_sh = c;
    if (do_v_shuffle) {
        const uint up_lane_u = (lane >= tx) ? (lane - tx) : lane;
        const uint dn_lane_u = ((lane + tx) < 32u) ? (lane + tx) : lane;
        u_sh = simd_shuffle(c, (ushort)up_lane_u);
        d_sh = simd_shuffle(c, (ushort)dn_lane_u);
    }

    if (i == 0u || j == 0u || i + 1u >= nx || j + 1u >= ny) {
        u_out[idx] = c;
        return;
    }

    float l = l_sh;
    if (lane == 0u || lid.x == 0u) {
        l = u_in[idx - 1u];
    }

    float r = r_sh;
    if (lane == 31u || (lid.x + 1u) >= tx) {
        r = u_in[idx + 1u];
    }

    float u = u_sh;
    if (!do_v_shuffle || lane < tx || lid.y == 0u) {
        u = u_in[idx - nx];
    }

    float d = d_sh;
    if (!do_v_shuffle || (lane + tx) >= 32u || (lid.y + 1u) >= tpg.y) {
        d = u_in[idx + nx];
    }

    const float sum = (l + r) + (u + d);
    u_out[idx] = c + alpha * (sum - 4.0f * c);
}
```

Result of previous attempt:
        256x256_50: correct, 0.76 ms, 34.7 GB/s (effective, 8 B/cell) (17.4% of 200 GB/s)
       512x512_100: correct, 3.43 ms, 61.1 GB/s (effective, 8 B/cell) (30.5% of 200 GB/s)
      1024x1024_50: correct, 3.84 ms, 109.2 GB/s (effective, 8 B/cell) (54.6% of 200 GB/s)
  score (gmean of fraction): 0.3071

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
- iter  4: compile=OK | correct=True | score=0.3900575907602331
- iter  5: compile=OK | correct=True | score=0.3070785537375746

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
