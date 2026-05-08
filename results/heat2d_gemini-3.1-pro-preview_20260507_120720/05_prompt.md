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
                      uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint j = gid.y;
    uint nx = NX;
    uint ny = NY;
    
    // Guard against out-of-bounds threads in padded dispatches
    if (i >= nx || j >= ny) return;

    // Evaluate single flat index for minimal address arithmetic
    uint idx = j * nx + i;

    // Dirichlet boundary condition: boundary cells retain original values.
    // Checked BEFORE loading neighbor cells to cluster loads for inner cells,
    // maximizing Memory Level Parallelism (MLP) and cache utilization.
    if (i == 0 || j == 0 || i == nx - 1 || j == ny - 1) {
        u_out[idx] = u_in[idx];
        return;
    }

    // Cluster all memory loads to ensure they issue as a single dense packet
    float c = u_in[idx];
    float l = u_in[idx - 1];
    float r = u_in[idx + 1];
    float t = u_in[idx - nx];
    float b = u_in[idx + nx];
    
    float a = alpha;
    
    // Group additions into a depth-2 tree to maximize Instruction Level Parallelism
    float sum = (l + r) + (t + b);
    
    // Mathematically rewrite expression to reduce critical path via Fused Multiply-Add
    // Original: c + a * (sum - 4*c)
    // Rewrite:  a * sum + c * (1 - 4a)
    float c_coeff = 1.0f - 4.0f * a;
    
    // FMA executes immediately after sum and c_coeff are ready
    u_out[idx] = fma(a, sum, c * c_coeff);
}
```

Result of previous attempt:
        256x256_50: correct, 0.78 ms, 33.7 GB/s (effective, 8 B/cell) (16.8% of 200 GB/s)
       512x512_100: correct, 1.90 ms, 110.5 GB/s (effective, 8 B/cell) (55.3% of 200 GB/s)
      1024x1024_50: correct, 2.75 ms, 152.7 GB/s (effective, 8 B/cell) (76.3% of 200 GB/s)
  score (gmean of fraction): 0.4141

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
        256x256_50: correct, 0.74 ms, 35.6 GB/s (effective, 8 B/cell) (17.8% of 200 GB/s)
       512x512_100: correct, 2.28 ms, 92.0 GB/s (effective, 8 B/cell) (46.0% of 200 GB/s)
      1024x1024_50: correct, 1.88 ms, 222.8 GB/s (effective, 8 B/cell) (111.4% of 200 GB/s)
  score (gmean of fraction): 0.4504

## History

- iter  0: compile=OK | correct=True | score=0.4503535203778002
- iter  1: compile=OK | correct=True | score=0.3597485724594344
- iter  2: compile=OK | correct=True | score=0.4015504678097701
- iter  3: compile=OK | correct=True | score=0.39531045899529954
- iter  4: compile=OK | correct=True | score=0.41408714517370904

## Stagnation notice

Your last 3 correct attempts all scored within 15% of
the incumbent without surpassing it. You are circling a local
optimum. STOP making incremental edits to the previous kernel.
Propose a STRUCTURALLY different approach — pick at least one of:

- Different memory layout (e.g. SoA <-> AoS; AA-pattern single-buffer
  in-place updates; pack scalars into a `float4` channel)
- Threadgroup-memory cooperative loads + `threadgroup_barrier` so a
  block of threads share one DRAM read across many compute steps
- SIMD-group cooperative reductions / shuffles via `simd_broadcast`,
  `simd_shuffle_*`, `simd_sum`, `simd_max`
- Multi-step / temporal kernel fusion: process several iterations
  per dispatch, holding state in threadgroup or register memory
- Persistent threadgroups (one threadgroup owns a tile across the
  entire run, eliminating per-step dispatch overhead)
- Mixed precision (fp16 inner loop with fp32 accumulation) where
  tolerance allows

A reworded version of the previous kernel will not break out of
this plateau.

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
