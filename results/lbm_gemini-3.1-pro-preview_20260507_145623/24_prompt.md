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

kernel void lbm_step(device const float *f_in   [[buffer(0)]],
                     device       float *f_out  [[buffer(1)]],
                     constant uint        &NX   [[buffer(2)]],
                     constant uint        &NY   [[buffer(3)]],
                     constant float       &tau  [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]])
{
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

    // PULL streaming reads
    float f[9];
    f[0] = f_in[idx];
    f[1] = f_in[N + idx + uint(im1)];
    f[2] = f_in[2 * N + idx + uint(jm1)];
    f[3] = f_in[3 * N + idx + uint(ip1)];
    f[4] = f_in[4 * N + idx + uint(jp1)];
    f[5] = f_in[5 * N + idx + uint(im1 + jm1)];
    f[6] = f_in[6 * N + idx + uint(ip1 + jm1)];
    f[7] = f_in[7 * N + idx + uint(ip1 + jp1)];
    f[8] = f_in[8 * N + idx + uint(im1 + jp1)];

    // Tree-reduction for moments
    float rho = (f[0] + f[1]) + (f[2] + f[3]) + (f[4] + f[5]) + (f[6] + f[7]) + f[8];
    float ux = (f[1] + f[5] + f[8]) - (f[3] + f[6] + f[7]);
    float uy = (f[2] + f[5] + f[6]) - (f[4] + f[7] + f[8]);

    float inv_rho = 1.0f / rho;
    ux *= inv_rho;
    uy *= inv_rho;

    // BGK collision setup
    float usq = ux * ux + uy * uy;
    float inv_tau_val = 1.0f / tau;
    float oma = 1.0f - inv_tau_val;

    float rt = rho * inv_tau_val;
    float term0 = 1.0f - 1.5f * usq;

    float w0_rt = (4.0f / 9.0f) * rt;
    float w1_rt = (1.0f / 9.0f) * rt;
    float w5_rt = (1.0f / 36.0f) * rt;

    // Symmetric velocity combinations
    float ux_sq = ux * ux;
    float uy_sq = uy * uy;
    float cu5 = ux + uy;
    float cu6 = -ux + uy;
    float cu5_sq = cu5 * cu5;
    float cu6_sq = cu6 * cu6;

    // Factored equilibrium terms
    float term_x = w1_rt * (term0 + 4.5f * ux_sq);
    float term_y = w1_rt * (term0 + 4.5f * uy_sq);
    float term_xy = w5_rt * (term0 + 4.5f * cu5_sq);
    float term_xmy = w5_rt * (term0 + 4.5f * cu6_sq);

    float ux_w1 = w1_rt * 3.0f * ux;
    float uy_w1 = w1_rt * 3.0f * uy;
    float cu5_w5 = w5_rt * 3.0f * cu5;
    float cu6_w5 = w5_rt * 3.0f * cu6;

    // Ordered writes to memory
    f_out[idx]         = f[0] * oma + w0_rt * term0;
    f_out[N + idx]     = f[1] * oma + term_x + ux_w1;
    f_out[2 * N + idx] = f[2] * oma + term_y + uy_w1;
    f_out[3 * N + idx] = f[3] * oma + term_x - ux_w1;
    f_out[4 * N + idx] = f[4] * oma + term_y - uy_w1;
    f_out[5 * N + idx] = f[5] * oma + term_xy + cu5_w5;
    f_out[6 * N + idx] = f[6] * oma + term_xmy + cu6_w5;
    f_out[7 * N + idx] = f[7] * oma + term_xy - cu5_w5;
    f_out[8 * N + idx] = f[8] * oma + term_xmy - cu6_w5;
}
```

Result of previous attempt:
          64x64_50: correct, 0.51 ms, 29.1 GB/s (effective, 72 B/cell) (14.5% of 200 GB/s)
       128x128_100: correct, 1.97 ms, 59.9 GB/s (effective, 72 B/cell) (29.9% of 200 GB/s)
       256x256_100: correct, 2.23 ms, 212.0 GB/s (effective, 72 B/cell) (106.0% of 200 GB/s)
  score (gmean of fraction): 0.3587

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

- iter 16: compile=FAIL | correct=False | score=N/A
- iter 17: compile=OK | correct=True | score=0.3566821793420694
- iter 18: compile=FAIL | correct=False | score=N/A
- iter 19: compile=OK | correct=True | score=0.3787669880949169
- iter 20: compile=OK | correct=True | score=0.48588262793695974
- iter 21: compile=OK | correct=True | score=0.41638508578188804
- iter 22: compile=OK | correct=True | score=0.3685905205413258
- iter 23: compile=OK | correct=True | score=0.358679804304519

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
