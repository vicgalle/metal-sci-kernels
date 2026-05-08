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
                     uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint j = gid.y;
    
    if (i >= NX || j >= NY) return;

    int inx = int(NX);
    int iny = int(NY);
    uint N = NX * NY;
    uint idx = j * NX + i;

    // Fast 1D periodic boundary offset calculation
    int im1 = (i == 0) ? (inx - 1) : -1;
    int ip1 = (i == NX - 1) ? (1 - inx) : 1;
    int jm1 = (j == 0) ? (inx * (iny - 1)) : -inx;
    int jp1 = (j == NY - 1) ? -(inx * (iny - 1)) : inx;

    // 1) Pull streaming: issue 9 independent loads
    float f0 = f_in[idx];
    float f1 = f_in[1 * N + idx + im1];
    float f2 = f_in[2 * N + idx + jm1];
    float f3 = f_in[3 * N + idx + ip1];
    float f4 = f_in[4 * N + idx + jp1];
    float f5 = f_in[5 * N + idx + im1 + jm1];
    float f6 = f_in[6 * N + idx + ip1 + jm1];
    float f7 = f_in[7 * N + idx + ip1 + jp1];
    float f8 = f_in[8 * N + idx + im1 + jp1];

    // 2) Moments
    float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;

    float ux = (f1 + f5 + f8) - (f3 + f6 + f7);
    float uy = (f2 + f5 + f6) - (f4 + f7 + f8);

    float inv_rho = 1.0f / rho;
    float u = ux * inv_rho;
    float v = uy * inv_rho;

    float usq = u * u + v * v;
    float term0 = 1.0f - 1.5f * usq;

    float r_w0 = rho * (4.0f / 9.0f);
    float r_w1 = rho * (1.0f / 9.0f);
    float r_w5 = rho * (1.0f / 36.0f);

    float inv_tau = 1.0f / tau;

    // 3) BGK Collision explicitly unrolled (maps to tight FMAs)
    float feq0 = r_w0 * term0;
    
    float cu1 = u;
    float feq1 = r_w1 * (term0 + cu1 * (3.0f + 4.5f * cu1));
    
    float cu2 = v;
    float feq2 = r_w1 * (term0 + cu2 * (3.0f + 4.5f * cu2));
    
    float cu3 = -u;
    float feq3 = r_w1 * (term0 + cu3 * (3.0f + 4.5f * cu3));
    
    float cu4 = -v;
    float feq4 = r_w1 * (term0 + cu4 * (3.0f + 4.5f * cu4));
    
    float cu5 = u + v;
    float feq5 = r_w5 * (term0 + cu5 * (3.0f + 4.5f * cu5));
    
    float cu6 = -u + v;
    float feq6 = r_w5 * (term0 + cu6 * (3.0f + 4.5f * cu6));
    
    float cu7 = -cu5; // saves an ADD
    float feq7 = r_w5 * (term0 + cu7 * (3.0f + 4.5f * cu7));
    
    float cu8 = -cu6; // saves an ADD
    float feq8 = r_w5 * (term0 + cu8 * (3.0f + 4.5f * cu8));

    // Write back
    f_out[idx]         = f0 - inv_tau * (f0 - feq0);
    f_out[1 * N + idx] = f1 - inv_tau * (f1 - feq1);
    f_out[2 * N + idx] = f2 - inv_tau * (f2 - feq2);
    f_out[3 * N + idx] = f3 - inv_tau * (f3 - feq3);
    f_out[4 * N + idx] = f4 - inv_tau * (f4 - feq4);
    f_out[5 * N + idx] = f5 - inv_tau * (f5 - feq5);
    f_out[6 * N + idx] = f6 - inv_tau * (f6 - feq6);
    f_out[7 * N + idx] = f7 - inv_tau * (f7 - feq7);
    f_out[8 * N + idx] = f8 - inv_tau * (f8 - feq8);
}
```

Result of previous attempt:
          64x64_50: correct, 0.50 ms, 29.6 GB/s (effective, 72 B/cell) (14.8% of 200 GB/s)
       128x128_100: correct, 2.01 ms, 58.7 GB/s (effective, 72 B/cell) (29.4% of 200 GB/s)
       256x256_100: correct, 2.26 ms, 209.0 GB/s (effective, 72 B/cell) (104.5% of 200 GB/s)
  score (gmean of fraction): 0.3567

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

- iter 10: compile=OK | correct=True | score=0.46232862357349447
- iter 11: compile=OK | correct=True | score=0.3909012300348845
- iter 12: compile=OK | correct=True | score=0.3624193042474191
- iter 13: compile=OK | correct=True | score=0.5530764161168673
- iter 14: compile=OK | correct=True | score=0.48881450123883946
- iter 15: compile=OK | correct=True | score=0.36496197947723824
- iter 16: compile=FAIL | correct=False | score=N/A
- iter 17: compile=OK | correct=True | score=0.3566821793420694

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
