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

kernel void heat_step(device const float * restrict u_in  [[buffer(0)]],
                      device       float * restrict u_out [[buffer(1)]],
                      constant uint      &NX    [[buffer(2)]],
                      constant uint      &NY    [[buffer(3)]],
                      constant float     &alpha [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]]) {
    uint i = gid.x;
    uint j = gid.y;
    
    if (i >= NX || j >= NY) return;

    // Load center value unconditionally for all valid threads to ensure full
    // SIMD participation during the shuffle operations.
    uint idx = j * NX + i;
    float c = u_in[idx];
    
    // Share center values horizontally across the SIMD group.
    float l_shuffle = simd_shuffle_up(c, 1);
    float r_shuffle = simd_shuffle_down(c, 1);
    
    // Share row indices to safely detect when SIMD lanes cross row boundaries.
    uint y_l = simd_shuffle_up(j, 1);
    uint y_r = simd_shuffle_down(j, 1);
    
    // Dirichlet boundary conditions: output unchanged center and exit.
    if (i == 0 || j == 0 || i == NX - 1 || j == NY - 1) {
        u_out[idx] = c;
        return;
    }
    
    // Determine left neighbor: use shuffle if the left lane is within the same row,
    // otherwise fallback to a memory load.
    float l;
    if (lane > 0 && y_l == j) {
        l = l_shuffle;
    } else {
        l = u_in[idx - 1];
    }
    
    // Determine right neighbor: use shuffle if the right lane is within the same row,
    // otherwise fallback to a memory load.
    float r;
    if (lane < 31 && y_r == j) {
        r = r_shuffle;
    } else {
        r = u_in[idx + 1];
    }

    // Vertical neighbors must still be loaded from the L1 cache.
    float d = u_in[idx - NX];
    float u = u_in[idx + NX];

    u_out[idx] = c + alpha * (l + r + d + u - 4.0f * c);
}
```

Result of previous attempt:
  COMPILE FAILED: Error Domain=MTLLibraryErrorDomain Code=3 "program_source:4:53: error: expected ')'
kernel void heat_step(device const float * restrict u_in  [[buffer(0)]],
                                                    ^
program_source:4:22: note: to match this '('
kernel void heat_step(device const float * restrict u_in  [[buffer(0)]],
                     ^
program_source:11:14: error: use of undeclared identifier 'gid'
    uint i = gid.x;
             ^
program_source:12:14: error: use of undeclared identifier 'gid'
    uint j = gid.y;
             ^
program_source:14:14: error: use of undeclared identifier 'NX'
    if (i >= NX || j >= NY) return;
             ^
program_source:14:25: error: use of undeclared identifier 'NY'
    if (i >= NX || j >= NY) return;
                        ^
program_source:18:20: error: use of undeclared identifier 'NX'
    uint idx = j * NX + i;
                   ^
program_source:19:15: error: use of undeclared identifier 'u_in'
    float c = u_in[idx];
              ^
program_source:30:34: error: use of undeclared identifier 'NX'
    if (i == 0 || j == 0 || i == NX - 1 || j == NY - 1) {
                                 ^
program_source:30:49: error: use of undeclared identifier 'NY'
    if (i == 0 || j == 0 || i == NX - 1 || j == NY - 1) {
                                                ^
program_source:31:9: error: use of undeclared identifier 'u_out'
        u_out[idx] = c;
        ^
program_source:38:9: error: use of undeclared identifier 'lane'
    if (lane > 0 && y_l == j) {
        ^
program_source:41:13: error: use of undeclared identifier 'u_in'
        l = u_in[idx - 1];
            ^
program_source:47:9: error: use of undeclared identifier 'lane'
    if (lane < 31 && y_r == j) {
        ^
program_source:50:13: error: use of undeclared identifier 'u_in'
        r = u_in[idx + 1];
            ^
program_source:54:15: error: use of undeclared identifier 'u_in'
    float d = u_in[idx - NX];
              ^
program_source:54:26: error: use of undeclared identifier 'NX'
    float d = u_in[idx - NX];
                         ^
program_source:55:15: error: use of undeclared identifier 'u_in'
    float u = u_in[idx + NX];
              ^
program_source:55:26: error: use of undeclared identifier 'NX'
    float u = u_in[idx + NX];
                         ^
program_source:57:5: error: use of undeclared identifier 'u_out'
    u_out[idx] = c + alpha * (l + r + d + u - 4.0f * c);
    ^
program_source:57:22: error: use of undeclared identifier 'alpha'
    u_out[idx] = c + alpha * (l + r + d + u - 4.0f * c);
                     ^
" UserInfo={NSLocalizedDescription=program_source:4:53: error: expected ')'
kernel void heat_step(device const float * restrict u_in  [[buffer(0)]],
                                                    ^
program_source:4:22: note: to match this '('
kernel void heat_step(device const float * restrict u_in  [[buffer(0)]],
                     ^
program_source:11:14: error: use of undeclared identifier 'gid'
    uint i = gid.x;
             ^
program_source:12:14: error: use of undeclared identifier 'gid'
    uint j = gid.y;
             ^
program_source:14:14: error: use of undeclared identifier 'NX'
    if (i >= NX || j >= NY) return;
             ^
program_source:14:25: error: use of undeclared identifier 'NY'
    if (i >= NX || j >= NY) return;
                        ^
program_source:18:20: error: use of undeclared identifier 'NX'
    uint idx = j * NX + i;
                   ^
program_source:19:15: error: use of undeclared identifier 'u_in'
    float c = u_in[idx];
              ^
program_source:30:34: error: use of undeclared identifier 'NX'
    if (i == 0 || j == 0 || i == NX - 1 || j == NY - 1) {
                                 ^
program_source:30:49: error: use of undeclared identifier 'NY'
    if (i == 0 || j == 0 || i == NX - 1 || j == NY - 1) {
                                                ^
program_source:31:9: error: use of undeclared identifier 'u_out'
        u_out[idx] = c;
        ^
program_source:38:9: error: use of undeclared identifier 'lane'
    if (lane > 0 && y_l == j) {
        ^
program_source:41:13: error: use of undeclared identifier 'u_in'
        l = u_in[idx - 1];
            ^
program_source:47:9: error: use of undeclared identifier 'lane'
    if (lane < 31 && y_r == j) {
        ^
program_source:50:13: error: use of undeclared identifier 'u_in'
        r = u_in[idx + 1];
            ^
program_source:54:15: error: use of undeclared identifier 'u_in'
    float d = u_in[idx - NX];
              ^
program_source:54:26: error: use of undeclared identifier 'NX'
    float d = u_in[idx - NX];
                         ^
program_source:55:15: error: use of undeclared identifier 'u_in'
    float u = u_in[idx + NX];
              ^
program_source:55:26: error: use of undeclared identifier 'NX'
    float u = u_in[idx + NX];
                         ^
program_source:57:5: error: use of undeclared identifier 'u_out'
    u_out[idx] = c + alpha * (l + r + d + u - 4.0f * c);
    ^
program_source:57:22: error: use of undeclared identifier 'alpha'
    u_out[idx] = c + alpha * (l + r + d + u - 4.0f * c);
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

- iter  0: compile=OK | correct=True | score=0.4503535203778002
- iter  1: compile=OK | correct=True | score=0.3597485724594344
- iter  2: compile=OK | correct=True | score=0.4015504678097701
- iter  3: compile=OK | correct=True | score=0.39531045899529954
- iter  4: compile=OK | correct=True | score=0.41408714517370904
- iter  5: compile=OK | correct=True | score=0.25366881587902923
- iter  6: compile=OK | correct=True | score=0.32506451943771747
- iter  7: compile=FAIL | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
