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
    if (i >= NX || j >= NY) return;

    uint N = NX * NY;
    
    // Fast periodic boundary wrapping without modulo
    uint x_m1 = (i == 0) ? NX - 1 : i - 1;
    uint x_p1 = (i + 1 == NX) ? 0 : i + 1;
    uint y_m1 = (j == 0) ? NY - 1 : j - 1;
    uint y_p1 = (j + 1 == NY) ? 0 : j + 1;

    uint base    = j * NX;
    uint base_m1 = y_m1 * NX;
    uint base_p1 = y_p1 * NX;

    // Fully unrolled pull streaming loads
    float f0 = f_in[base + i];
    float f1 = f_in[base + x_m1 + N];
    float f2 = f_in[base_m1 + i + 2 * N];
    float f3 = f_in[base + x_p1 + 3 * N];
    float f4 = f_in[base_p1 + i + 4 * N];
    float f5 = f_in[base_m1 + x_m1 + 5 * N];
    float f6 = f_in[base_m1 + x_p1 + 6 * N];
    float f7 = f_in[base_p1 + x_p1 + 7 * N];
    float f8 = f_in[base_p1 + x_m1 + 8 * N];

    // Macroscopic moments
    float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    float ux = f1 - f3 + f5 - f6 - f7 + f8;
    float uy = f2 - f4 + f5 + f6 - f7 - f8;

    float inv_rho = 1.0f / rho;
    ux *= inv_rho;
    uy *= inv_rho;

    // BGK Collision
    float usq = ux * ux + uy * uy;
    float usq_term = 1.0f - 1.5f * usq;
    float inv_tau = 1.0f / tau;
    
    float w_rho_0 = (4.0f / 9.0f) * rho;
    float w_rho_1 = (1.0f / 9.0f) * rho;
    float w_rho_5 = (1.0f / 36.0f) * rho;

    float feq0 = w_rho_0 * usq_term;
    float out0 = f0 - inv_tau * (f0 - feq0);

    float cu1 = ux;
    float feq1 = w_rho_1 * (usq_term + cu1 * (3.0f + 4.5f * cu1));
    float out1 = f1 - inv_tau * (f1 - feq1);

    float cu2 = uy;
    float feq2 = w_rho_1 * (usq_term + cu2 * (3.0f + 4.5f * cu2));
    float out2 = f2 - inv_tau * (f2 - feq2);

    float cu3 = -ux;
    float feq3 = w_rho_1 * (usq_term + cu3 * (3.0f + 4.5f * cu3));
    float out3 = f3 - inv_tau * (f3 - feq3);

    float cu4 = -uy;
    float feq4 = w_rho_1 * (usq_term + cu4 * (3.0f + 4.5f * cu4));
    float out4 = f4 - inv_tau * (f4 - feq4);

    float cu5 = ux + uy;
    float feq5 = w_rho_5 * (usq_term + cu5 * (3.0f + 4.5f * cu5));
    float out5 = f5 - inv_tau * (f5 - feq5);

    float cu6 = -ux + uy;
    float feq6 = w_rho_5 * (usq_term + cu6 * (3.0f + 4.5f * cu6));
    float out6 = f6 - inv_tau * (f6 - feq6);

    float cu7 = -cu5;
    float feq7 = w_rho_5 * (usq_term + cu7 * (3.0f + 4.5f * cu7));
    float out7 = f7 - inv_tau * (f7 - feq7);

    float cu8 = -cu6;
    float feq8 = w_rho_5 * (usq_term + cu8 * (3.0f + 4.5f * cu8));
    float out8 = f8 - inv_tau * (f8 - feq8);

    // Fully unrolled stores
    uint out_idx = base + i;
    f_out[out_idx]         = out0;
    f_out[out_idx + N]     = out1;
    f_out[out_idx + 2 * N] = out2;
    f_out[out_idx + 3 * N] = out3;
    f_out[out_idx + 4 * N] = out4;
    f_out[out_idx + 5 * N] = out5;
    f_out[out_idx + 6 * N] = out6;
    f_out[out_idx + 7 * N] = out7;
    f_out[out_idx + 8 * N] = out8;
}
```

Result of previous attempt:
          64x64_50: correct, 0.53 ms, 27.8 GB/s (effective, 72 B/cell) (13.9% of 200 GB/s)
       128x128_100: correct, 1.09 ms, 108.0 GB/s (effective, 72 B/cell) (54.0% of 200 GB/s)
       256x256_100: correct, 2.19 ms, 215.1 GB/s (effective, 72 B/cell) (107.5% of 200 GB/s)
  score (gmean of fraction): 0.4321

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

- iter  0: compile=OK | correct=True | score=0.5198830134549006
- iter  1: compile=OK | correct=True | score=0.34978854643190943
- iter  2: compile=OK | correct=True | score=0.35825317794817724
- iter  3: compile=OK | correct=True | score=0.4703949429262635
- iter  4: compile=OK | correct=True | score=0.3825859708218299
- iter  5: compile=OK | correct=True | score=0.35137941309385284
- iter  6: compile=OK | correct=True | score=0.4429690944125417
- iter  7: compile=OK | correct=True | score=0.43205974293706484

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
