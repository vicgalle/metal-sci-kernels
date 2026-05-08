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
                      uint2 gid [[thread_position_in_grid]]) 
{
    uint nx = NX;
    uint ny = NY;
    
    uint i = gid.x;
    uint j = gid.y;
    
    // Evaluate validity but do NOT return early yet so all lanes can participate in the shuffles.
    // Invalid threads are safely pointed to idx 0 to prevent out-of-bounds memory accesses.
    bool valid = (i < nx && j < ny);
    uint idx = valid ? (j * nx + i) : 0;
    
    // Every thread loads its center value (invalid threads load u_in[0])
    float c = u_in[idx];
    
    // --- SIMDgroup Cooperation Phase ---
    uint lane = simd_lane_id;
    uint width = simd_execution_width;
    
    // Shuffle the linear index to robustly identify if adjacent lanes hold the true spatial neighbor
    uint l_idx = simd_shuffle_up(idx, 1);
    uint r_idx = simd_shuffle_down(idx, 1);
    
    // Shuffle the data values 
    float l_shfl = simd_shuffle_up(c, 1);
    float r_shfl = simd_shuffle_down(c, 1);
    
    // Now that shuffles are complete, invalid threads safely exit
    if (!valid) return;
    
    // Dirichlet boundary conditions: edge cells stay at their initial value.
    if (i == 0 || j == 0 || i == nx - 1 || j == ny - 1) {
        u_out[idx] = c;
        return;
    }
    
    // --- Internal Cells ONLY ---
    // Fetch horizontal neighbors from registers if available, else fallback to L1 cache.
    // Using explicit if/else ensures that the fallback load is strictly predicated.
    float l;
    if (lane > 0 && l_idx == idx - 1) {
        l = l_shfl;
    } else {
        l = u_in[idx - 1];
    }
    
    float r;
    if (lane < width - 1 && r_idx == idx + 1) {
        r = r_shfl;
    } else {
        r = u_in[idx + 1];
    }
    
    // Vertical neighbors must still be loaded from L1 cache (different rows).
    float d = u_in[idx - nx];
    float u = u_in[idx + nx];
    
    // FMA optimization with grouped additions to maximize instruction-level parallelism
    u_out[idx] = fma(alpha, (l + r) + (d + u) - 4.0f * c, c);
}
```

Result of previous attempt:
  COMPILE FAILED: Error Domain=MTLLibraryErrorDomain Code=3 "program_source:26:17: error: use of undeclared identifier 'simd_lane_id'
    uint lane = simd_lane_id;
                ^
program_source:27:18: error: use of undeclared identifier 'simd_execution_width'
    uint width = simd_execution_width;
                 ^
" UserInfo={NSLocalizedDescription=program_source:26:17: error: use of undeclared identifier 'simd_lane_id'
    uint lane = simd_lane_id;
                ^
program_source:27:18: error: use of undeclared identifier 'simd_execution_width'
    uint width = simd_execution_width;
                 ^
}

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

- iter  2: compile=OK | correct=True | score=0.4015504678097701
- iter  3: compile=OK | correct=True | score=0.39531045899529954
- iter  4: compile=OK | correct=True | score=0.41408714517370904
- iter  5: compile=OK | correct=True | score=0.25366881587902923
- iter  6: compile=OK | correct=True | score=0.32506451943771747
- iter  7: compile=FAIL | correct=False | score=N/A
- iter  8: compile=OK | correct=True | score=0.3709370073735379
- iter  9: compile=FAIL | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
