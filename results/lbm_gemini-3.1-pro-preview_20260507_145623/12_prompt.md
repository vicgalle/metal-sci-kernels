## Task: lbm

D2Q9 lattice Boltzmann method, fused pull-streaming + BGK collision, periodic boundary conditions. Distribution functions are stored SoA: f[k * NX*NY + j*NX + i] for k in [0, 9), j in [0, NY), i in [0, NX), float32 row-major.

Per timestep, per cell (i, j):
  1) PULL stream: f_streamed[k] = f_in[k, (i - CX[k]) mod NX,
                                          (j - CY[k]) mod NY]
  2) Moments: rho = sum_k f_streamed[k];
     u = (sum_k CX[k] * f_streamed[k]) / rho; v likewise.
  3) BGK collision: f_out[k] = f_streamed[k]
       - (1/tau) (f_streamed[k] - f_eq[k])
     with f_eq[k] = W[k] * rho *
       (1 + 3 (CX[k] u + CY[k] v)
          + 4.5 (CX[k] u + CY[k] v)^2 - 1.5 (u^2 + v^2)).
Velocity table CX[9] = {0, 1, 0,-1, 0, 1,-1,-1, 1};
                CY[9] = {0, 0, 1, 0,-1, 1, 1,-1,-1};
weights W[9] = {4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36}.

The host runs the kernel n_steps times with two buffers ping-ponged each call. Effective DRAM traffic per step is 72 bytes/cell (9 reads + 9 writes), so the roofline is BW-bound.

## Required kernel signature(s)

```
kernel void lbm_step(device const float *f_in   [[buffer(0)]],
                     device       float *f_out  [[buffer(1)]],
                     constant uint        &NX   [[buffer(2)]],
                     constant uint        &NY   [[buffer(3)]],
                     constant float       &tau  [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]]);

Grid is dispatched 2-D as `threadsPerGrid = (NX, NY)`, one thread per output cell — guard with `if (i >= NX || j >= NY) return;`. Each thread MUST update exactly one cell; the host will not shrink the dispatch if you process multiple cells per thread, so extra threads just idle. SoA layout MUST be preserved on buffers 0 and 1; the kernel may use any internal layout/optimization (threadgroup tiling, simdgroup ops, etc.).
```

## Your previous attempt

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant float CX[9] = {0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 1.0f, -1.0f, -1.0f, 1.0f};
constexpr constant float CY[9] = {0.0f, 0.0f, 1.0f, 0.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f};
constexpr constant float W[9] = {
    4.0f / 9.0f,
    1.0f / 9.0f,  1.0f / 9.0f,  1.0f / 9.0f,  1.0f / 9.0f,
    1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f
};

kernel void lbm_step(device const float *f_in   [[buffer(0)]],
                     device       float *f_out  [[buffer(1)]],
                     constant uint        &NX   [[buffer(2)]],
                     constant uint        &NY   [[buffer(3)]],
                     constant float       &tau  [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint j = gid.y;
    uint nx = NX;
    uint ny = NY;
    
    if (i >= nx || j >= ny) return;

    uint N = nx * ny;

    // Fast periodic boundary wrapping using select instead of modulo division
    uint im1 = (i > 0) ? i - 1 : nx - 1;
    uint ip1 = (i == nx - 1) ? 0 : i + 1;
    uint jm1 = (j > 0) ? j - 1 : ny - 1;
    uint jp1 = (j == ny - 1) ? 0 : j + 1;

    uint j_nx   = j * nx;
    uint jm1_nx = jm1 * nx;
    uint jp1_nx = jp1 * nx;

    // Precompute perfectly aligned 1D offsets for pull streaming
    uint idx_src[9];
    idx_src[0] = j_nx   + i;
    idx_src[1] = j_nx   + im1;
    idx_src[2] = jm1_nx + i;
    idx_src[3] = j_nx   + ip1;
    idx_src[4] = jp1_nx + i;
    idx_src[5] = jm1_nx + im1;
    idx_src[6] = jm1_nx + ip1;
    idx_src[7] = jp1_nx + ip1;
    idx_src[8] = jp1_nx + im1;

    float f[9];
    float rho = 0.0f;
    float ux  = 0.0f;
    float uy  = 0.0f;

    // Load streamed values and accumulate macroscopic moments
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        float val = f_in[k * N + idx_src[k]];
        f[k] = val;
        rho += val;
        ux  += CX[k] * val;
        uy  += CY[k] * val;
    }

    float inv_rho = 1.0f / rho;
    ux *= inv_rho;
    uy *= inv_rho;

    // BGK collision optimized with Horner's method for the polynomial
    float usq = ux * ux + uy * uy;
    float usq_term = 1.0f - 1.5f * usq;
    float inv_tau = 1.0f / tau;
    uint idx = j_nx + i;

    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        float cu = CX[k] * ux + CY[k] * uy;
        float feq = W[k] * rho * (usq_term + cu * (3.0f + 4.5f * cu));
        f_out[k * N + idx] = f[k] - inv_tau * (f[k] - feq);
    }
}
```

Result of previous attempt:
          64x64_50: correct, 0.48 ms, 30.7 GB/s (effective, 72 B/cell) (15.3% of 200 GB/s)
       128x128_100: correct, 2.05 ms, 57.5 GB/s (effective, 72 B/cell) (28.8% of 200 GB/s)
       256x256_100: correct, 1.74 ms, 270.7 GB/s (effective, 72 B/cell) (135.4% of 200 GB/s)
  score (gmean of fraction): 0.3909

## Current best (incumbent)

```metal
// Naive seed kernel for D2Q9 lattice Boltzmann (BGK collision + pull
// streaming, periodic boundary conditions).
//
// Per timestep, per cell:
//   1. Pull: gather f_in[k] from upstream cell (i - cx[k], j - cy[k]) with
//      periodic wrap.
//   2. Moments: rho = sum_k f_k; rho*u = sum_k c_k f_k.
//   3. BGK collision: f_out[k] = f_streamed[k] - (1/tau) (f_streamed[k] - f_eq_k)
//      with f_eq_k = w_k rho (1 + 3 c_k.u + 9/2 (c_k.u)^2 - 3/2 |u|^2).
//
// Storage is structure-of-arrays (SoA): f[k * NX*NY + j*NX + i] for
// k in [0, 9). Channels are contiguous; this is the layout used by
// Schoenherr 2011 et al. and is the natural baseline for cache-efficient
// vector loads.
//
// Buffer layout (kept stable across the candidate; the LLM may experiment
// with internal layouts but must read buffer 0 and write buffer 1 in this
// SoA convention):
//   buffer 0: const float *f_in  (length 9 * NX * NY)
//   buffer 1: device float *f_out (length 9 * NX * NY)
//   buffer 2: const uint &NX
//   buffer 3: const uint &NY
//   buffer 4: const float &tau   (relaxation time, ~0.6..2.0)

#include <metal_stdlib>
using namespace metal;

// Velocity directions:
//   0: ( 0, 0)   1: (+1, 0)   2: ( 0,+1)   3: (-1, 0)   4: ( 0,-1)
//   5: (+1,+1)   6: (-1,+1)   7: (-1,-1)   8: (+1,-1)
constant int CX[9] = {0,  1,  0, -1,  0,  1, -1, -1,  1};
constant int CY[9] = {0,  0,  1,  0, -1,  1,  1, -1, -1};
constant float W[9] = {
    4.0f / 9.0f,
    1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
    1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f
};

kernel void lbm_step(device const float *f_in   [[buffer(0)]],
                     device       float *f_out  [[buffer(1)]],
                     constant uint        &NX   [[buffer(2)]],
                     constant uint        &NY   [[buffer(3)]],
                     constant float       &tau  [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint j = gid.y;
    if (i >= NX || j >= NY) return;

    uint N    = NX * NY;
    int  inx  = int(NX);
    int  iny  = int(NY);

    // 1. Pull streaming with periodic wrap.
    float f[9];
    for (int k = 0; k < 9; ++k) {
        int srcx = int(i) - CX[k];
        int srcy = int(j) - CY[k];
        srcx = (srcx + inx) % inx;
        srcy = (srcy + iny) % iny;
        f[k] = f_in[uint(k) * N + uint(srcy) * NX + uint(srcx)];
    }

    // 2. Moments.
    float rho = 0.0f;
    float ux  = 0.0f;
    float uy  = 0.0f;
    for (int k = 0; k < 9; ++k) {
        rho += f[k];
        ux  += float(CX[k]) * f[k];
        uy  += float(CY[k]) * f[k];
    }
    float inv_rho = 1.0f / rho;
    ux *= inv_rho;
    uy *= inv_rho;

    // 3. BGK collision.
    float usq = ux * ux + uy * uy;
    float inv_tau = 1.0f / tau;
    uint  idx = j * NX + i;
    for (int k = 0; k < 9; ++k) {
        float cu  = float(CX[k]) * ux + float(CY[k]) * uy;
        float feq = W[k] * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * usq);
        f_out[uint(k) * N + idx] = f[k] - inv_tau * (f[k] - feq);
    }
}
```

Incumbent result:
          64x64_50: correct, 0.51 ms, 28.9 GB/s (effective, 72 B/cell) (14.5% of 200 GB/s)
       128x128_100: correct, 0.73 ms, 161.5 GB/s (effective, 72 B/cell) (80.8% of 200 GB/s)
       256x256_100: correct, 1.96 ms, 240.6 GB/s (effective, 72 B/cell) (120.3% of 200 GB/s)
  score (gmean of fraction): 0.5199

## History

- iter  4: compile=OK | correct=True | score=0.3825859708218299
- iter  5: compile=OK | correct=True | score=0.35137941309385284
- iter  6: compile=OK | correct=True | score=0.4429690944125417
- iter  7: compile=OK | correct=True | score=0.43205974293706484
- iter  8: compile=OK | correct=True | score=0.3497938458825147
- iter  9: compile=FAIL | correct=False | score=N/A
- iter 10: compile=OK | correct=True | score=0.46232862357349447
- iter 11: compile=OK | correct=True | score=0.3909012300348845

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
