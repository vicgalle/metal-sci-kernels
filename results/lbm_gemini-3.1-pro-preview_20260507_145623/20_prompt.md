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
    int N = inx * iny;
    int idx = int(j) * inx + int(i);

    // Fast branchless periodic boundary offset calculation
    int im1 = select(-1, inx - 1, i == 0);
    int ip1 = select(1, 1 - inx, i == nx - 1);
    int jm1 = select(-inx, inx * (iny - 1), j == 0);
    int jp1 = select(inx, -inx * (iny - 1), j == ny - 1);

    device const float *p_in = f_in + idx;
    device       float *p_out = f_out + idx;

    // Fully unrolled pull streaming
    float f0 = p_in[0];
    float f1 = p_in[N + im1];
    float f2 = p_in[2 * N + jm1];
    float f3 = p_in[3 * N + ip1];
    float f4 = p_in[4 * N + jp1];
    float f5 = p_in[5 * N + im1 + jm1];
    float f6 = p_in[6 * N + ip1 + jm1];
    float f7 = p_in[7 * N + ip1 + jp1];
    float f8 = p_in[8 * N + im1 + jp1];

    // Macroscopic moments with instruction-level parallelism
    float f56_sum = f5 + f6;
    float f78_sum = f7 + f8;
    float f56_diff = f5 - f6;
    float f78_diff = f7 - f8;

    float rho = f0 + (f1 + f2) + (f3 + f4) + f56_sum + f78_sum;
    float ux = (f1 - f3) + f56_diff - f78_diff;
    float uy = (f2 - f4) + f56_sum - f78_sum;

    float inv_rho = 1.0f / rho;
    ux *= inv_rho;
    uy *= inv_rho;

    // Algebraic simplification for BGK equilibrium sharing shared factors
    float ux_sq = ux * ux;
    float uy_sq = uy * uy;
    float usq = ux_sq + uy_sq;
    float term0 = fma(-1.5f, usq, 1.0f);

    // Straight directional terms
    float ux3 = 3.0f * ux;
    float uy3 = 3.0f * uy;

    float term_x = fma(4.5f, ux_sq, term0);
    float poly1 = term_x + ux3;
    float poly3 = term_x - ux3;

    float term_y = fma(4.5f, uy_sq, term0);
    float poly2 = term_y + uy3;
    float poly4 = term_y - uy3;

    // Diagonal directional terms
    float term_diag = fma(3.0f, usq, 1.0f);
    float ux_uy_9 = 9.0f * ux * uy;
    
    float term_p = term_diag + ux_uy_9;
    float term_m = term_diag - ux_uy_9;

    float up3 = ux3 + uy3;
    float um3 = ux3 - uy3;

    float poly5 = term_p + up3;
    float poly7 = term_p - up3;
    float poly8 = term_m + um3;
    float poly6 = term_m - um3;

    // BGK Collision execution mapping uniformly to chains
    float inv_tau = 1.0f / tau;
    float oma = 1.0f - inv_tau;
    float rho_tau = rho * inv_tau;

    constexpr float W0 = 4.0f / 9.0f;
    constexpr float W1 = 1.0f / 9.0f;
    constexpr float W5 = 1.0f / 36.0f;

    float w0_rho_tau = W0 * rho_tau;
    float w1_rho_tau = W1 * rho_tau;
    float w5_rho_tau = W5 * rho_tau;

    p_out[0]     = fma(f0, oma, w0_rho_tau * term0);
    p_out[N]     = fma(f1, oma, w1_rho_tau * poly1);
    p_out[2 * N] = fma(f2, oma, w1_rho_tau * poly2);
    p_out[3 * N] = fma(f3, oma, w1_rho_tau * poly3);
    p_out[4 * N] = fma(f4, oma, w1_rho_tau * poly4);
    p_out[5 * N] = fma(f5, oma, w5_rho_tau * poly5);
    p_out[6 * N] = fma(f6, oma, w5_rho_tau * poly6);
    p_out[7 * N] = fma(f7, oma, w5_rho_tau * poly7);
    p_out[8 * N] = fma(f8, oma, w5_rho_tau * poly8);
}
```

Result of previous attempt:
          64x64_50: correct, 0.46 ms, 32.3 GB/s (effective, 72 B/cell) (16.1% of 200 GB/s)
       128x128_100: correct, 1.97 ms, 60.0 GB/s (effective, 72 B/cell) (30.0% of 200 GB/s)
       256x256_100: correct, 2.10 ms, 224.3 GB/s (effective, 72 B/cell) (112.2% of 200 GB/s)
  score (gmean of fraction): 0.3788

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

- iter 12: compile=OK | correct=True | score=0.3624193042474191
- iter 13: compile=OK | correct=True | score=0.5530764161168673
- iter 14: compile=OK | correct=True | score=0.48881450123883946
- iter 15: compile=OK | correct=True | score=0.36496197947723824
- iter 16: compile=FAIL | correct=False | score=N/A
- iter 17: compile=OK | correct=True | score=0.3566821793420694
- iter 18: compile=FAIL | correct=False | score=N/A
- iter 19: compile=OK | correct=True | score=0.3787669880949169

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
