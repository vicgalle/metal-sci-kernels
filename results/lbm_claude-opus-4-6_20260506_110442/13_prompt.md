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
    const uint i = gid.x;
    const uint j = gid.y;
    if (i >= NX || j >= NY) return;

    const uint N = NX * NY;
    const float inv_tau = 1.0f / tau;
    const float omtau = 1.0f - inv_tau;

    // Periodic neighbors - branchless
    const uint im1 = (i > 0u) ? (i - 1u) : (NX - 1u);
    const uint ip1 = (i + 1u < NX) ? (i + 1u) : 0u;
    const uint jm1 = (j > 0u) ? (j - 1u) : (NY - 1u);
    const uint jp1 = (j + 1u < NY) ? (j + 1u) : 0u;

    // Precompute row offsets
    const uint rj   = j   * NX;
    const uint rjm1 = jm1 * NX;
    const uint rjp1 = jp1 * NX;

    // Pull streaming: gather from source cells
    // Spread reads across different k-planes to maximize memory-level parallelism
    const float f0 = f_in[          rj   + i  ];
    const float f5 = f_in[5u * N + rjm1 + im1];
    const float f1 = f_in[    N  + rj   + im1];
    const float f6 = f_in[6u * N + rjm1 + ip1];
    const float f2 = f_in[2u * N + rjm1 + i  ];
    const float f7 = f_in[7u * N + rjp1 + ip1];
    const float f3 = f_in[3u * N + rj   + ip1];
    const float f8 = f_in[8u * N + rjp1 + im1];
    const float f4 = f_in[4u * N + rjp1 + i  ];

    // Moments with maximum ILP - compute partial sums in parallel chains
    const float s13 = f1 + f3;
    const float d13 = f1 - f3;
    const float s24 = f2 + f4;
    const float d24 = f2 - f4;
    const float s58 = f5 + f8;
    const float d58 = f5 - f8;  // unused but helps structure
    const float s67 = f6 + f7;
    const float d67 = f6 - f7;

    const float rho = f0 + s13 + s24 + s58 + s67;
    const float inv_rho = 1.0f / rho;

    // ux = (f1 - f3 + f5 - f6 - f7 + f8) / rho
    const float ux = (d13 + (f5 + f8) - (f6 + f7)) * inv_rho;
    // uy = (f2 - f4 + f5 + f6 - f7 - f8) / rho
    const float uy = (d24 + (f5 + f6) - (f7 + f8)) * inv_rho;

    // Precompute collision terms
    const float ux2 = ux * ux;
    const float uy2 = uy * uy;
    const float usq15 = 1.5f * (ux2 + uy2);
    const float base = 1.0f - usq15;

    const float ux3 = 3.0f * ux;
    const float uy3 = 3.0f * uy;

    // Pre-multiply weight * rho * inv_tau for each weight class
    const float rw49_t  = (4.0f / 9.0f)  * rho * inv_tau;
    const float rw19_t  = (1.0f / 9.0f)  * rho * inv_tau;
    const float rw136_t = (1.0f / 36.0f) * rho * inv_tau;

    const uint idx = rj + i;

    // k=0: cu=0
    f_out[idx] = fma(omtau, f0, rw49_t * base);

    // k=1: cu=ux, k=3: cu=-ux
    {
        const float ux45 = 4.5f * ux2;
        const float sym = base + ux45;
        f_out[    N + idx] = fma(omtau, f1, rw19_t * (sym + ux3));
        f_out[3u* N + idx] = fma(omtau, f3, rw19_t * (sym - ux3));
    }

    // k=2: cu=uy, k=4: cu=-uy
    {
        const float uy45 = 4.5f * uy2;
        const float sym = base + uy45;
        f_out[2u* N + idx] = fma(omtau, f2, rw19_t * (sym + uy3));
        f_out[4u* N + idx] = fma(omtau, f4, rw19_t * (sym - uy3));
    }

    // k=5: cu=ux+uy, k=7: cu=-(ux+uy)
    {
        const float cu = ux + uy;
        const float cu3 = ux3 + uy3;
        const float sym = fma(4.5f, cu * cu, base);
        f_out[5u* N + idx] = fma(omtau, f5, rw136_t * (sym + cu3));
        f_out[7u* N + idx] = fma(omtau, f7, rw136_t * (sym - cu3));
    }

    // k=6: cu=uy-ux, k=8: cu=-(uy-ux)
    {
        const float cu = uy - ux;
        const float cu3 = uy3 - ux3;
        const float sym = fma(4.5f, cu * cu, base);
        f_out[6u* N + idx] = fma(omtau, f6, rw136_t * (sym + cu3));
        f_out[8u* N + idx] = fma(omtau, f8, rw136_t * (sym - cu3));
    }
}
```

Result of previous attempt:
          64x64_50: correct, 0.47 ms, 31.3 GB/s (effective, 72 B/cell) (15.7% of 200 GB/s)
       128x128_100: correct, 2.02 ms, 58.4 GB/s (effective, 72 B/cell) (29.2% of 200 GB/s)
       256x256_100: correct, 2.19 ms, 215.2 GB/s (effective, 72 B/cell) (107.6% of 200 GB/s)
  score (gmean of fraction): 0.3665

## Current best (incumbent)

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
    const uint i = gid.x;
    const uint j = gid.y;
    if (i >= NX || j >= NY) return;

    const uint N = NX * NY;
    const float inv_tau = 1.0f / tau;
    const float omtau = 1.0f - inv_tau;

    // Periodic neighbors - branchless using ternary
    const uint im1 = (i > 0u) ? (i - 1u) : (NX - 1u);
    const uint ip1 = (i + 1u < NX) ? (i + 1u) : 0u;
    const uint jm1 = (j > 0u) ? (j - 1u) : (NY - 1u);
    const uint jp1 = (j + 1u < NY) ? (j + 1u) : 0u;

    // Precompute row offsets
    const uint rj   = j   * NX;
    const uint rjm1 = jm1 * NX;
    const uint rjp1 = jp1 * NX;

    // Pull streaming with careful ordering to spread across memory banks
    const float f0 = f_in[          rj   + i  ];
    const float f1 = f_in[    N   + rj   + im1];
    const float f2 = f_in[2u * N + rjm1 + i  ];
    const float f3 = f_in[3u * N + rj   + ip1];
    const float f4 = f_in[4u * N + rjp1 + i  ];
    const float f5 = f_in[5u * N + rjm1 + im1];
    const float f6 = f_in[6u * N + rjm1 + ip1];
    const float f7 = f_in[7u * N + rjp1 + ip1];
    const float f8 = f_in[8u * N + rjp1 + im1];

    // Moments - structured for ILP
    const float rho = f0 + (f1 + f3) + (f2 + f4) + (f5 + f6) + (f7 + f8);
    const float inv_rho = 1.0f / rho;

    // ux = (f1 - f3 + f5 - f6 - f7 + f8) / rho
    const float ux = ((f1 - f3) + (f5 - f6) + (f8 - f7)) * inv_rho;
    // uy = (f2 - f4 + f5 + f6 - f7 - f8) / rho
    const float uy = ((f2 - f4) + (f5 - f7) + (f6 - f8)) * inv_rho;

    // Precompute collision terms
    const float ux2 = ux * ux;
    const float uy2 = uy * uy;
    const float usq15 = 1.5f * (ux2 + uy2);
    const float base = 1.0f - usq15;

    const float ux3 = 3.0f * ux;
    const float uy3 = 3.0f * uy;

    // Weighted rho
    const float rw49  = (4.0f / 9.0f)  * rho;
    const float rw19  = (1.0f / 9.0f)  * rho;
    const float rw136 = (1.0f / 36.0f) * rho;

    const uint idx = rj + i;

    // k=0: cu=0
    {
        const float feq = rw49 * base;
        f_out[idx] = fma(omtau, f0, inv_tau * feq);
    }

    // k=1,3: cu = +/-ux
    {
        const float ux45 = 4.5f * ux2;
        const float sym = base + ux45;
        const float rw = rw19 * inv_tau;
        const float feq1 = rw * fma(1.0f, ux3, sym);
        const float feq3 = rw * (sym - ux3);
        f_out[N + idx]      = fma(omtau, f1, feq1);
        f_out[3u * N + idx] = fma(omtau, f3, feq3);
    }

    // k=2,4: cu = +/-uy
    {
        const float uy45 = 4.5f * uy2;
        const float sym = base + uy45;
        const float rw = rw19 * inv_tau;
        const float feq2 = rw * fma(1.0f, uy3, sym);
        const float feq4 = rw * (sym - uy3);
        f_out[2u * N + idx] = fma(omtau, f2, feq2);
        f_out[4u * N + idx] = fma(omtau, f4, feq4);
    }

    // k=5,7: cu = +/-(ux+uy)
    {
        const float cu = ux + uy;
        const float cu3 = ux3 + uy3;
        const float sym = fma(4.5f, cu * cu, base);
        const float rw = rw136 * inv_tau;
        f_out[5u * N + idx] = fma(omtau, f5, rw * (sym + cu3));
        f_out[7u * N + idx] = fma(omtau, f7, rw * (sym - cu3));
    }

    // k=6,8: cu = +/-(uy-ux)
    {
        const float cu = uy - ux;
        const float cu3 = uy3 - ux3;
        const float sym = fma(4.5f, cu * cu, base);
        const float rw = rw136 * inv_tau;
        f_out[6u * N + idx] = fma(omtau, f6, rw * (sym + cu3));
        f_out[8u * N + idx] = fma(omtau, f8, rw * (sym - cu3));
    }
}
```

Incumbent result:
          64x64_50: correct, 0.26 ms, 57.5 GB/s (effective, 72 B/cell) (28.7% of 200 GB/s)
       128x128_100: correct, 1.19 ms, 99.1 GB/s (effective, 72 B/cell) (49.6% of 200 GB/s)
       256x256_100: correct, 2.08 ms, 226.8 GB/s (effective, 72 B/cell) (113.4% of 200 GB/s)
  score (gmean of fraction): 0.5447

## History

- iter  5: compile=OK | correct=True | score=0.45069447223334436
- iter  6: compile=OK | correct=True | score=0.47899409212905025
- iter  7: compile=OK | correct=True | score=0.5446546034528802
- iter  8: compile=OK | correct=True | score=0.36589960630590984
- iter  9: compile=FAIL | correct=False | score=N/A
- iter 10: compile=OK | correct=True | score=0.377166344357743
- iter 11: compile=OK | correct=True | score=0.48243871624063395
- iter 12: compile=OK | correct=True | score=0.3665105508144934

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
