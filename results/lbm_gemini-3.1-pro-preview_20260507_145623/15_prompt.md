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

[[max_total_threads_per_threadgroup(256)]]
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

    int inx = int(nx);
    int iny = int(ny);
    int N = inx * iny;
    int idx = int(j * nx + i);

    int im1_val = inx - 1;
    int jm1_val = N - inx;
    
    int im1 = (i == 0) ? im1_val : -1;
    int ip1 = (i == nx - 1) ? -im1_val : 1;
    int jm1 = (j == 0) ? jm1_val : -inx;
    int jp1 = (j == ny - 1) ? -jm1_val : inx;

    // Pull streaming reads
    float f0 = f_in[idx];
    float f1 = f_in[1 * N + idx + im1];
    float f2 = f_in[2 * N + idx + jm1];
    float f3 = f_in[3 * N + idx + ip1];
    float f4 = f_in[4 * N + idx + jp1];
    float f5 = f_in[5 * N + idx + im1 + jm1];
    float f6 = f_in[6 * N + idx + ip1 + jm1];
    float f7 = f_in[7 * N + idx + ip1 + jp1];
    float f8 = f_in[8 * N + idx + im1 + jp1];

    // Symmetries for moments
    float f1_f3 = f1 - f3;
    float f5_f7 = f5 - f7;
    float f8_f6 = f8 - f6;
    float f1_p_f3 = f1 + f3;
    float f2_p_f4 = f2 + f4;
    float f5_p_f7 = f5 + f7;
    float f6_p_f8 = f6 + f8;

    float rho = f0 + f1_p_f3 + f2_p_f4 + f5_p_f7 + f6_p_f8;
    float inv_rho = 1.0f / rho;
    
    float ux = (f1_f3 + f5_f7 + f8_f6) * inv_rho;
    float uy = ((f2 - f4) + f5_f7 - f8_f6) * inv_rho;

    // Precompute constants for BGK
    float cu1_sq = ux * ux;
    float cu2_sq = uy * uy;
    float usq = cu1_sq + cu2_sq;
    float om_usq15 = 1.0f - 1.5f * usq;

    float w9_rho = rho * (1.0f / 9.0f);
    float w36_rho = rho * (1.0f / 36.0f);

    float ux3 = 3.0f * ux;
    float uy3 = 3.0f * uy;

    float base1 = om_usq15 + 4.5f * cu1_sq;
    float feq1 = w9_rho * (base1 + ux3);
    float feq3 = w9_rho * (base1 - ux3);

    float base2 = om_usq15 + 4.5f * cu2_sq;
    float feq2 = w9_rho * (base2 + uy3);
    float feq4 = w9_rho * (base2 - uy3);

    float ux_uy_9 = 9.0f * (ux * uy);
    float base56 = 1.0f + 3.0f * usq;
    float base5 = base56 + ux_uy_9;
    float base6 = base56 - ux_uy_9;

    float uxy3_plus = ux3 + uy3;
    float uxy3_minus = uy3 - ux3;

    float feq5 = w36_rho * (base5 + uxy3_plus);
    float feq7 = w36_rho * (base5 - uxy3_plus);

    float feq6 = w36_rho * (base6 + uxy3_minus);
    float feq8 = w36_rho * (base6 - uxy3_minus);

    float feq0 = rho * (4.0f / 9.0f) * om_usq15;

    float inv_tau = 1.0f / tau;

    // Writes
    f_out[idx] = f0 + inv_tau * (feq0 - f0);
    f_out[1 * N + idx] = f1 + inv_tau * (feq1 - f1);
    f_out[2 * N + idx] = f2 + inv_tau * (feq2 - f2);
    f_out[3 * N + idx] = f3 + inv_tau * (feq3 - f3);
    f_out[4 * N + idx] = f4 + inv_tau * (feq4 - f4);
    f_out[5 * N + idx] = f5 + inv_tau * (feq5 - f5);
    f_out[6 * N + idx] = f6 + inv_tau * (feq6 - f6);
    f_out[7 * N + idx] = f7 + inv_tau * (feq7 - f7);
    f_out[8 * N + idx] = f8 + inv_tau * (feq8 - f8);
}
```

Result of previous attempt:
          64x64_50: correct, 0.25 ms, 58.4 GB/s (effective, 72 B/cell) (29.2% of 200 GB/s)
       128x128_100: correct, 1.63 ms, 72.2 GB/s (effective, 72 B/cell) (36.1% of 200 GB/s)
       256x256_100: correct, 2.13 ms, 221.8 GB/s (effective, 72 B/cell) (110.9% of 200 GB/s)
  score (gmean of fraction): 0.4888

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

constant float CX[9] = {0.0f,  1.0f,  0.0f, -1.0f,  0.0f,  1.0f, -1.0f, -1.0f,  1.0f};
constant float CY[9] = {0.0f,  0.0f,  1.0f,  0.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f};
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
    uint nx = NX;
    uint ny = NY;
    
    if (i >= nx || j >= ny) return;

    int inx = int(nx);
    int iny = int(ny);
    uint N = nx * ny;
    uint idx = j * nx + i;

    // Fast 1D periodic boundary offset calculation
    int im1 = (i == 0) ? (inx - 1) : -1;
    int ip1 = (i == nx - 1) ? (1 - inx) : 1;
    int jm1 = (j == 0) ? (inx * (iny - 1)) : -inx;
    int jp1 = (j == ny - 1) ? -(inx * (iny - 1)) : inx;

    const int off[9] = {
        0,
        im1,
        jm1,
        ip1,
        jp1,
        im1 + jm1,
        ip1 + jm1,
        ip1 + jp1,
        im1 + jp1
    };

    float f[9];
    float rho = 0.0f;
    float ux = 0.0f;
    float uy = 0.0f;

    // Pull streaming and moments computation
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        float v = f_in[k * N + idx + uint(off[k])];
        f[k] = v;
        rho += v;
        ux += CX[k] * v;
        uy += CY[k] * v;
    }

    float inv_rho = 1.0f / rho;
    ux *= inv_rho;
    uy *= inv_rho;

    float usq = ux * ux + uy * uy;
    float inv_tau = 1.0f / tau;

    // BGK collision
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        float cu = CX[k] * ux + CY[k] * uy;
        float feq = W[k] * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * usq);
        f_out[k * N + idx] = f[k] - inv_tau * (f[k] - feq);
    }
}
```

Incumbent result:
          64x64_50: correct, 0.23 ms, 64.4 GB/s (effective, 72 B/cell) (32.2% of 200 GB/s)
       128x128_100: correct, 1.15 ms, 102.4 GB/s (effective, 72 B/cell) (51.2% of 200 GB/s)
       256x256_100: correct, 2.30 ms, 205.1 GB/s (effective, 72 B/cell) (102.6% of 200 GB/s)
  score (gmean of fraction): 0.5531

## History

- iter  7: compile=OK | correct=True | score=0.43205974293706484
- iter  8: compile=OK | correct=True | score=0.3497938458825147
- iter  9: compile=FAIL | correct=False | score=N/A
- iter 10: compile=OK | correct=True | score=0.46232862357349447
- iter 11: compile=OK | correct=True | score=0.3909012300348845
- iter 12: compile=OK | correct=True | score=0.3624193042474191
- iter 13: compile=OK | correct=True | score=0.5530764161168673
- iter 14: compile=OK | correct=True | score=0.48881450123883946

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
