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

    // Fast periodic boundary wrapping (eliminates expensive modulo)
    uint im1 = (i == 0) ? NX - 1 : i - 1;
    uint ip1 = (i == NX - 1) ? 0 : i + 1;
    uint jm1 = (j == 0) ? NY - 1 : j - 1;
    uint jp1 = (j == NY - 1) ? 0 : j + 1;

    // Precalculate row offsets
    uint row_j   = j * NX;
    uint row_jm1 = jm1 * NX;
    uint row_jp1 = jp1 * NX;

    // 1. Pull streaming phase
    // Explicit scalarization guarantees zero local memory allocation and perfectly maps to registers.
    float f0 = f_in[0 * N + row_j   + i  ];
    float f1 = f_in[1 * N + row_j   + im1];
    float f2 = f_in[2 * N + row_jm1 + i  ];
    float f3 = f_in[3 * N + row_j   + ip1];
    float f4 = f_in[4 * N + row_jp1 + i  ];
    float f5 = f_in[5 * N + row_jm1 + im1];
    float f6 = f_in[6 * N + row_jm1 + ip1];
    float f7 = f_in[7 * N + row_jp1 + ip1];
    float f8 = f_in[8 * N + row_jp1 + im1];

    // 2. Macroscopic moments
    // Sequentially accumulated to strictly preserve bitwise math tolerance.
    float rho = 0.0f;
    rho += f0; rho += f1; rho += f2;
    rho += f3; rho += f4; rho += f5;
    rho += f6; rho += f7; rho += f8;

    float ux = 0.0f;
    ux += f1; ux -= f3; ux += f5;
    ux -= f6; ux -= f7; ux += f8;

    float uy = 0.0f;
    uy += f2; uy -= f4; uy += f5;
    uy += f6; uy -= f7; uy -= f8;

    float inv_rho = 1.0f / rho;
    ux *= inv_rho;
    uy *= inv_rho;

    // 3. BGK collision & Write phase
    float usq = ux * ux + uy * uy;
    float inv_tau = 1.0f / tau;
    
    // Evaluate equilibrium polynomial components optimally
    float base = 1.0f - 1.5f * usq;

    float w0 = 4.0f / 9.0f;
    float w1 = 1.0f / 9.0f;
    float w5 = 1.0f / 36.0f;

    float rho_w0 = rho * w0;
    float rho_w1 = rho * w1;
    float rho_w5 = rho * w5;

    uint idx = row_j + i;

    // Interleave operations to map into immediate stores, reducing live register pressure
    float feq0 = rho_w0 * base;
    f_out[0 * N + idx] = f0 - inv_tau * (f0 - feq0);

    float cu1 = ux;
    float feq1 = rho_w1 * (base + cu1 * (3.0f + 4.5f * cu1));
    f_out[1 * N + idx] = f1 - inv_tau * (f1 - feq1);

    float cu2 = uy;
    float feq2 = rho_w1 * (base + cu2 * (3.0f + 4.5f * cu2));
    f_out[2 * N + idx] = f2 - inv_tau * (f2 - feq2);

    float cu3 = -ux;
    float feq3 = rho_w1 * (base + cu3 * (3.0f + 4.5f * cu3));
    f_out[3 * N + idx] = f3 - inv_tau * (f3 - feq3);

    float cu4 = -uy;
    float feq4 = rho_w1 * (base + cu4 * (3.0f + 4.5f * cu4));
    f_out[4 * N + idx] = f4 - inv_tau * (f4 - feq4);

    float cu5 = ux + uy;
    float feq5 = rho_w5 * (base + cu5 * (3.0f + 4.5f * cu5));
    f_out[5 * N + idx] = f5 - inv_tau * (f5 - feq5);

    float cu6 = -ux + uy;
    float feq6 = rho_w5 * (base + cu6 * (3.0f + 4.5f * cu6));
    f_out[6 * N + idx] = f6 - inv_tau * (f6 - feq6);

    // Reuse inverted directional momenta
    float cu7 = -cu5;
    float feq7 = rho_w5 * (base + cu7 * (3.0f + 4.5f * cu7));
    f_out[7 * N + idx] = f7 - inv_tau * (f7 - feq7);

    float cu8 = -cu6;
    float feq8 = rho_w5 * (base + cu8 * (3.0f + 4.5f * cu8));
    f_out[8 * N + idx] = f8 - inv_tau * (f8 - feq8);
}
```

Result of previous attempt:
          64x64_50: correct, 0.51 ms, 29.0 GB/s (effective, 72 B/cell) (14.5% of 200 GB/s)
       128x128_100: correct, 2.04 ms, 57.9 GB/s (effective, 72 B/cell) (28.9% of 200 GB/s)
       256x256_100: correct, 2.28 ms, 206.6 GB/s (effective, 72 B/cell) (103.3% of 200 GB/s)
  score (gmean of fraction): 0.3514

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

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
