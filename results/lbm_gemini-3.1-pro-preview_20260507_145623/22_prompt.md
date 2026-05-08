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

// Exploit symmetry, optimize BGK math, and hint max threadgroup size 
// to ensure sufficient threadgroups to fill GPU cores on small grids.
[[max_total_threads_per_threadgroup(256)]]
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

    uint N = nx * ny;
    uint idx = j * nx + i;

    // Fast 1D purely unsigned periodic boundary offset calculation.
    // Relies on 32-bit unsigned wrapping to natively avoid branching / modulo operators.
    uint im1 = (i > 0) ? -1u : (nx - 1);
    uint ip1 = (i < nx - 1) ? 1u : (1u - nx);
    uint jm1 = (j > 0) ? -nx : (nx * (ny - 1));
    uint jp1 = (j < ny - 1) ? nx : -(nx * (ny - 1));

    // Pull streamed values from neighbors
    float f0 = f_in[idx];
    float f1 = f_in[N + idx + im1];
    float f2 = f_in[2 * N + idx + jm1];
    float f3 = f_in[3 * N + idx + ip1];
    float f4 = f_in[4 * N + idx + jp1];
    float f5 = f_in[5 * N + idx + im1 + jm1];
    float f6 = f_in[6 * N + idx + ip1 + jm1];
    float f7 = f_in[7 * N + idx + ip1 + jp1];
    float f8 = f_in[8 * N + idx + im1 + jp1];

    // Compute macroscopic moments with optimized instruction grouping
    float f5p6 = f5 + f6;
    float f5m6 = f5 - f6;
    float f7p8 = f7 + f8;
    float f7m8 = f7 - f8;

    float rho = f0 + (f1 + f2) + (f3 + f4) + (f5p6 + f7p8);
    float ux  = (f1 - f3) + (f5m6 - f7m8);
    float uy  = (f2 - f4) + (f5p6 - f7p8);

    float inv_rho = 1.0f / rho;
    ux *= inv_rho;
    uy *= inv_rho;

    // Common terms for BGK collision
    float usq = ux * ux + uy * uy;
    float term0 = 1.0f - 1.5f * usq;

    float inv_tau_val = 1.0f / tau;
    float oma = 1.0f - inv_tau_val;
    float rt = rho * inv_tau_val;

    float w0_rt = (4.0f / 9.0f) * rt;
    float w1_rt = (1.0f / 9.0f) * rt;
    float w5_rt = (1.0f / 36.0f) * rt;

    // Execute collision. Exploit symmetry: opposite directions share the squared term.
    
    // k = 0
    f_out[idx] = f0 * oma + w0_rt * term0;

    // k = 1, 3
    float ux_sq = ux * ux;
    float term_x = term0 + 4.5f * ux_sq;
    float ux3 = 3.0f * ux;
    f_out[N + idx]     = f1 * oma + w1_rt * (term_x + ux3);
    f_out[3 * N + idx] = f3 * oma + w1_rt * (term_x - ux3);

    // k = 2, 4
    float uy_sq = uy * uy;
    float term_y = term0 + 4.5f * uy_sq;
    float uy3 = 3.0f * uy;
    f_out[2 * N + idx] = f2 * oma + w1_rt * (term_y + uy3);
    f_out[4 * N + idx] = f4 * oma + w1_rt * (term_y - uy3);

    // k = 5, 7
    float cu5 = ux + uy;
    float term_xy = term0 + 4.5f * (cu5 * cu5);
    float cu5_3 = 3.0f * cu5;
    f_out[5 * N + idx] = f5 * oma + w5_rt * (term_xy + cu5_3);
    f_out[7 * N + idx] = f7 * oma + w5_rt * (term_xy - cu5_3);

    // k = 6, 8
    float cu6 = -ux + uy;
    float term_xmy = term0 + 4.5f * (cu6 * cu6);
    float cu6_3 = 3.0f * cu6;
    f_out[6 * N + idx] = f6 * oma + w5_rt * (term_xmy + cu6_3);
    f_out[8 * N + idx] = f8 * oma + w5_rt * (term_xmy - cu6_3);
}
```

Result of previous attempt:
          64x64_50: correct, 0.36 ms, 41.4 GB/s (effective, 72 B/cell) (20.7% of 200 GB/s)
       128x128_100: correct, 1.92 ms, 61.5 GB/s (effective, 72 B/cell) (30.7% of 200 GB/s)
       256x256_100: correct, 2.08 ms, 226.8 GB/s (effective, 72 B/cell) (113.4% of 200 GB/s)
  score (gmean of fraction): 0.4164

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

- iter 14: compile=OK | correct=True | score=0.48881450123883946
- iter 15: compile=OK | correct=True | score=0.36496197947723824
- iter 16: compile=FAIL | correct=False | score=N/A
- iter 17: compile=OK | correct=True | score=0.3566821793420694
- iter 18: compile=FAIL | correct=False | score=N/A
- iter 19: compile=OK | correct=True | score=0.3787669880949169
- iter 20: compile=OK | correct=True | score=0.48588262793695974
- iter 21: compile=OK | correct=True | score=0.41638508578188804

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
