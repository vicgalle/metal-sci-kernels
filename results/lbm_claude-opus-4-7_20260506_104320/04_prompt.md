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

Grid is dispatched 2-D as (NX, NY); guard with `if (i >= NX || j >= NY) return;`. SoA layout MUST be preserved on buffers 0 and 1; the kernel may use any internal layout/optimization.
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
    uint nx = NX;
    uint ny = NY;
    if (i >= nx || j >= ny) return;

    uint N  = nx * ny;

    uint ip1 = (i + 1u == nx) ? 0u : (i + 1u);
    uint im1 = (i == 0u) ? (nx - 1u) : (i - 1u);
    uint jp1 = (j + 1u == ny) ? 0u : (j + 1u);
    uint jm1 = (j == 0u) ? (ny - 1u) : (j - 1u);

    uint row    = j   * nx;
    uint rowU   = jm1 * nx;
    uint rowD   = jp1 * nx;

    float f0 = f_in[0u * N + row  + i  ];
    float f1 = f_in[1u * N + row  + im1];
    float f2 = f_in[2u * N + rowU + i  ];
    float f3 = f_in[3u * N + row  + ip1];
    float f4 = f_in[4u * N + rowD + i  ];
    float f5 = f_in[5u * N + rowU + im1];
    float f6 = f_in[6u * N + rowU + ip1];
    float f7 = f_in[7u * N + rowD + ip1];
    float f8 = f_in[8u * N + rowD + im1];

    float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    float jx  = (f1 - f3) + (f5 - f6) + (f8 - f7);
    float jy  = (f2 - f4) + (f5 + f6) - (f7 + f8);
    float inv_rho = 1.0f / rho;
    float ux = jx * inv_rho;
    float uy = jy * inv_rho;

    float usq = ux * ux + uy * uy;
    float omega = 1.0f / tau;
    float one_minus_omega = 1.0f - omega;

    const float W0 = 4.0f / 9.0f;
    const float W1 = 1.0f / 9.0f;
    const float W5 = 1.0f / 36.0f;

    float c1 = 1.0f - 1.5f * usq;

    // Pre-scaled weights: omega * W * rho
    float wr0 = omega * W0 * rho;
    float wr1 = omega * W1 * rho;
    float wr5 = omega * W5 * rho;

    uint idx = row + i;

    // k=0
    f_out[0u * N + idx] = fma(one_minus_omega, f0, wr0 * c1);

    // k=1: cu = ux
    {
        float cu = ux;
        float poly = fma(4.5f * cu, cu, fma(3.0f, cu, c1));
        f_out[1u * N + idx] = fma(one_minus_omega, f1, wr1 * poly);
    }
    // k=2: cu = uy
    {
        float cu = uy;
        float poly = fma(4.5f * cu, cu, fma(3.0f, cu, c1));
        f_out[2u * N + idx] = fma(one_minus_omega, f2, wr1 * poly);
    }
    // k=3: cu = -ux
    {
        float cu = -ux;
        float poly = fma(4.5f * cu, cu, fma(3.0f, cu, c1));
        f_out[3u * N + idx] = fma(one_minus_omega, f3, wr1 * poly);
    }
    // k=4: cu = -uy
    {
        float cu = -uy;
        float poly = fma(4.5f * cu, cu, fma(3.0f, cu, c1));
        f_out[4u * N + idx] = fma(one_minus_omega, f4, wr1 * poly);
    }
    // k=5: cu = ux + uy
    {
        float cu = ux + uy;
        float poly = fma(4.5f * cu, cu, fma(3.0f, cu, c1));
        f_out[5u * N + idx] = fma(one_minus_omega, f5, wr5 * poly);
    }
    // k=6: cu = -ux + uy
    {
        float cu = uy - ux;
        float poly = fma(4.5f * cu, cu, fma(3.0f, cu, c1));
        f_out[6u * N + idx] = fma(one_minus_omega, f6, wr5 * poly);
    }
    // k=7: cu = -ux - uy
    {
        float cu = -(ux + uy);
        float poly = fma(4.5f * cu, cu, fma(3.0f, cu, c1));
        f_out[7u * N + idx] = fma(one_minus_omega, f7, wr5 * poly);
    }
    // k=8: cu = ux - uy
    {
        float cu = ux - uy;
        float poly = fma(4.5f * cu, cu, fma(3.0f, cu, c1));
        f_out[8u * N + idx] = fma(one_minus_omega, f8, wr5 * poly);
    }
}
```

Result of previous attempt:
          64x64_50: correct, 0.48 ms, 30.5 GB/s (effective, 72 B/cell) (15.2% of 200 GB/s)
       128x128_100: correct, 0.90 ms, 131.2 GB/s (effective, 72 B/cell) (65.6% of 200 GB/s)
       256x256_100: correct, 2.26 ms, 209.0 GB/s (effective, 72 B/cell) (104.5% of 200 GB/s)
  score (gmean of fraction): 0.4709

## Current best (incumbent)

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

    uint nx = NX;
    uint ny = NY;
    uint N  = nx * ny;

    // Branchless periodic wrap for neighbor coordinates.
    uint ip1 = (i + 1u == nx) ? 0u : (i + 1u);
    uint im1 = (i == 0u) ? (nx - 1u) : (i - 1u);
    uint jp1 = (j + 1u == ny) ? 0u : (j + 1u);
    uint jm1 = (j == 0u) ? (ny - 1u) : (j - 1u);

    uint row    = j   * nx;
    uint rowU   = jm1 * nx;  // upstream for cy=+1 (source j-1)
    uint rowD   = jp1 * nx;  // upstream for cy=-1 (source j+1)

    // Pull-stream: f[k] = f_in[k, (i-cx) mod NX, (j-cy) mod NY]
    // k=0: (0,0)    -> (i, j)
    // k=1: (+1,0)   -> (i-1, j)
    // k=2: (0,+1)   -> (i, j-1)
    // k=3: (-1,0)   -> (i+1, j)
    // k=4: (0,-1)   -> (i, j+1)
    // k=5: (+1,+1)  -> (i-1, j-1)
    // k=6: (-1,+1)  -> (i+1, j-1)
    // k=7: (-1,-1)  -> (i+1, j+1)
    // k=8: (+1,-1)  -> (i-1, j+1)
    float f0 = f_in[0u * N + row  + i  ];
    float f1 = f_in[1u * N + row  + im1];
    float f2 = f_in[2u * N + rowU + i  ];
    float f3 = f_in[3u * N + row  + ip1];
    float f4 = f_in[4u * N + rowD + i  ];
    float f5 = f_in[5u * N + rowU + im1];
    float f6 = f_in[6u * N + rowU + ip1];
    float f7 = f_in[7u * N + rowD + ip1];
    float f8 = f_in[8u * N + rowD + im1];

    // Moments.
    float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    float jx  = (f1 - f3) + (f5 - f6) + (f8 - f7);
    float jy  = (f2 - f4) + (f5 + f6) - (f7 + f8);
    float inv_rho = 1.0f / rho;
    float ux = jx * inv_rho;
    float uy = jy * inv_rho;

    float usq = ux * ux + uy * uy;
    float inv_tau = 1.0f / tau;
    float omega = inv_tau;
    float one_minus_omega = 1.0f - omega;

    const float W0 = 4.0f / 9.0f;
    const float W1 = 1.0f / 9.0f;
    const float W5 = 1.0f / 36.0f;

    float k15 = 1.5f * usq;
    float c1  = 1.0f - k15;

    // Helper: feq_k = W_k * rho * (1 + 3 cu + 4.5 cu^2 - 1.5 usq)
    // Compute cu for each direction.
    float cu;
    float feq;
    uint idx = row + i;

    // k=0: cu = 0
    feq = W0 * rho * c1;
    f_out[0u * N + idx] = one_minus_omega * f0 + omega * feq;

    // k=1: cu = ux
    cu = ux;
    feq = W1 * rho * (c1 + 3.0f * cu + 4.5f * cu * cu);
    f_out[1u * N + idx] = one_minus_omega * f1 + omega * feq;

    // k=2: cu = uy
    cu = uy;
    feq = W1 * rho * (c1 + 3.0f * cu + 4.5f * cu * cu);
    f_out[2u * N + idx] = one_minus_omega * f2 + omega * feq;

    // k=3: cu = -ux
    cu = -ux;
    feq = W1 * rho * (c1 + 3.0f * cu + 4.5f * cu * cu);
    f_out[3u * N + idx] = one_minus_omega * f3 + omega * feq;

    // k=4: cu = -uy
    cu = -uy;
    feq = W1 * rho * (c1 + 3.0f * cu + 4.5f * cu * cu);
    f_out[4u * N + idx] = one_minus_omega * f4 + omega * feq;

    // k=5: cu = ux + uy
    cu = ux + uy;
    feq = W5 * rho * (c1 + 3.0f * cu + 4.5f * cu * cu);
    f_out[5u * N + idx] = one_minus_omega * f5 + omega * feq;

    // k=6: cu = -ux + uy
    cu = -ux + uy;
    feq = W5 * rho * (c1 + 3.0f * cu + 4.5f * cu * cu);
    f_out[6u * N + idx] = one_minus_omega * f6 + omega * feq;

    // k=7: cu = -ux - uy
    cu = -ux - uy;
    feq = W5 * rho * (c1 + 3.0f * cu + 4.5f * cu * cu);
    f_out[7u * N + idx] = one_minus_omega * f7 + omega * feq;

    // k=8: cu = ux - uy
    cu = ux - uy;
    feq = W5 * rho * (c1 + 3.0f * cu + 4.5f * cu * cu);
    f_out[8u * N + idx] = one_minus_omega * f8 + omega * feq;
}
```

Incumbent result:
          64x64_50: correct, 0.22 ms, 66.6 GB/s (effective, 72 B/cell) (33.3% of 200 GB/s)
       128x128_100: correct, 1.27 ms, 93.0 GB/s (effective, 72 B/cell) (46.5% of 200 GB/s)
       256x256_100: correct, 2.11 ms, 223.3 GB/s (effective, 72 B/cell) (111.7% of 200 GB/s)
  score (gmean of fraction): 0.5571

## History

- iter  0: compile=OK | correct=True | score=0.4845923223348039
- iter  1: compile=OK | correct=True | score=0.5571141820858078
- iter  2: compile=OK | correct=True | score=0.388747055363346
- iter  3: compile=OK | correct=True | score=0.4708963926910348

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
