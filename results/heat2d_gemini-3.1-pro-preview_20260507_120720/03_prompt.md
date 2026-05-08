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

[[max_total_threads_per_threadgroup(1024)]]
kernel void heat_step(device const float *u_in  [[buffer(0)]],
                      device       float *u_out [[buffer(1)]],
                      constant uint      &NX_ref    [[buffer(2)]],
                      constant uint      &NY_ref    [[buffer(3)]],
                      constant float     &alpha_ref [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint j = gid.y;
    uint NX = NX_ref;
    uint NY = NY_ref;
    
    // Bounds check for the grid dispatch padding
    if (i >= NX || j >= NY) return;

    // Single flat index for minimal address arithmetic
    uint idx = j * NX + i;

    // Read center element unconditionally to hide latency
    float c = u_in[idx];

    // Dirichlet boundary condition: boundary cells stay at their initial value.
    // Uses integer underflow to combine (i == 0 || i == NX - 1) into a single check.
    if ((i - 1) >= (NX - 2) || (j - 1) >= (NY - 2)) {
        u_out[idx] = c;
        return;
    }

    // Load neighbors (perfectly cached by L1)
    float l = u_in[idx - 1];
    float r = u_in[idx + 1];
    float u = u_in[idx - NX];
    float d = u_in[idx + NX];
    
    float alpha = alpha_ref;
    
    // Group neighbor additions into a depth-2 tree for ILP
    float sum = (l + r) + (u + d);
    
    // Mathematically rewrite `c + alpha * (sum - 4*c)` to remove one FP operation
    // from the critical path. The uniform value `1 - 4*alpha` is hoisted by the compiler.
    float c_coeff = 1.0f - 4.0f * alpha;
    
    // Fused multiply-add for optimal throughput and accuracy
    u_out[idx] = fma(alpha, sum, c * c_coeff);
}
```

Result of previous attempt:
        256x256_50: correct, 0.77 ms, 34.2 GB/s (effective, 8 B/cell) (17.1% of 200 GB/s)
       512x512_100: correct, 1.91 ms, 110.1 GB/s (effective, 8 B/cell) (55.0% of 200 GB/s)
      1024x1024_50: correct, 3.05 ms, 137.7 GB/s (effective, 8 B/cell) (68.9% of 200 GB/s)
  score (gmean of fraction): 0.4016

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

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
