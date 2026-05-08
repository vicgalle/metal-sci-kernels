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
    if (i >= NX || j >= NY) return;

    uint N  = NX * NY;
    uint nx = NX;
    uint ny = NY;

    // Branchless periodic neighbors (offsets are only -1, 0, +1).
    uint im1 = (i == 0u)        ? (nx - 1u) : (i - 1u);
    uint ip1 = (i == nx - 1u)   ? 0u        : (i + 1u);
    uint jm1 = (j == 0u)        ? (ny - 1u) : (j - 1u);
    uint jp1 = (j == ny - 1u)   ? 0u        : (j + 1u);

    uint row    = j   * nx;
    uint rowm1  = jm1 * nx;
    uint rowp1  = jp1 * nx;

    // Pull streaming: source = (i - CX[k], j - CY[k]) mod (NX, NY).
    // CX = {0, 1, 0,-1, 0, 1,-1,-1, 1};
    // CY = {0, 0, 1, 0,-1, 1, 1,-1,-1};
    float f0 = f_in[0u * N + row   + i  ];
    float f1 = f_in[1u * N + row   + im1];
    float f2 = f_in[2u * N + rowm1 + i  ];
    float f3 = f_in[3u * N + row   + ip1];
    float f4 = f_in[4u * N + rowp1 + i  ];
    float f5 = f_in[5u * N + rowm1 + im1];
    float f6 = f_in[6u * N + rowm1 + ip1];
    float f7 = f_in[7u * N + rowp1 + ip1];
    float f8 = f_in[8u * N + rowp1 + im1];

    // Moments.
    float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    float jx  = (f1 - f3) + (f5 - f6) - (f7 - f8); // = f1 - f3 + f5 - f6 - f7 + f8
    float jy  = (f2 - f4) + (f5 + f6) - (f7 + f8); // = f2 - f4 + f5 + f6 - f7 - f8

    float inv_rho = 1.0f / rho;
    float ux = jx * inv_rho;
    float uy = jy * inv_rho;

    float usq    = ux * ux + uy * uy;
    float inv_tau = 1.0f / tau;
    float omega   = inv_tau;
    float one_m_omega = 1.0f - omega;

    // Equilibrium prefactors.
    const float W0 = 4.0f / 9.0f;
    const float W1 = 1.0f / 9.0f;
    const float W5 = 1.0f / 36.0f;

    float rW0 = rho * W0;
    float rW1 = rho * W1;
    float rW5 = rho * W5;

    float k1 = 1.0f - 1.5f * usq;

    // For each k: feq = W[k]*rho*(1 + 3 cu + 4.5 cu^2 - 1.5 usq)
    // f_out = (1-omega)*f + omega*feq

    // k=0: cu = 0
    float feq0 = rW0 * k1;
    f_out[0u * N + row + i] = one_m_omega * f0 + omega * feq0;

    // k=1: cu = ux
    {
        float cu = ux;
        float feq = rW1 * (k1 + 3.0f * cu + 4.5f * cu * cu);
        f_out[1u * N + row + i] = one_m_omega * f1 + omega * feq;
    }
    // k=2: cu = uy
    {
        float cu = uy;
        float feq = rW1 * (k1 + 3.0f * cu + 4.5f * cu * cu);
        f_out[2u * N + row + i] = one_m_omega * f2 + omega * feq;
    }
    // k=3: cu = -ux
    {
        float cu = -ux;
        float feq = rW1 * (k1 + 3.0f * cu + 4.5f * cu * cu);
        f_out[3u * N + row + i] = one_m_omega * f3 + omega * feq;
    }
    // k=4: cu = -uy
    {
        float cu = -uy;
        float feq = rW1 * (k1 + 3.0f * cu + 4.5f * cu * cu);
        f_out[4u * N + row + i] = one_m_omega * f4 + omega * feq;
    }
    // k=5: cu = ux + uy
    {
        float cu = ux + uy;
        float feq = rW5 * (k1 + 3.0f * cu + 4.5f * cu * cu);
        f_out[5u * N + row + i] = one_m_omega * f5 + omega * feq;
    }
    // k=6: cu = -ux + uy
    {
        float cu = -ux + uy;
        float feq = rW5 * (k1 + 3.0f * cu + 4.5f * cu * cu);
        f_out[6u * N + row + i] = one_m_omega * f6 + omega * feq;
    }
    // k=7: cu = -ux - uy
    {
        float cu = -ux - uy;
        float feq = rW5 * (k1 + 3.0f * cu + 4.5f * cu * cu);
        f_out[7u * N + row + i] = one_m_omega * f7 + omega * feq;
    }
    // k=8: cu = ux - uy
    {
        float cu = ux - uy;
        float feq = rW5 * (k1 + 3.0f * cu + 4.5f * cu * cu);
        f_out[8u * N + row + i] = one_m_omega * f8 + omega * feq;
    }
}
```

Result of previous attempt:
          64x64_50: correct, 0.47 ms, 31.7 GB/s (effective, 72 B/cell) (15.8% of 200 GB/s)
       128x128_100: correct, 1.98 ms, 59.6 GB/s (effective, 72 B/cell) (29.8% of 200 GB/s)
       256x256_100: correct, 1.68 ms, 280.8 GB/s (effective, 72 B/cell) (140.4% of 200 GB/s)
  score (gmean of fraction): 0.4048

## History

- iter  0: compile=OK | correct=True | score=0.3777021884080966
- iter  1: compile=OK | correct=True | score=0.40481474566612136

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
