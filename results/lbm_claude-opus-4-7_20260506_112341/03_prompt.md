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

    uint N  = NX * NY;

    // Branchless periodic neighbors for ±1.
    uint im1 = select(i - 1u, NX - 1u, i == 0u);
    uint ip1 = select(i + 1u, 0u,      i + 1u == NX);
    uint jm1 = select(j - 1u, NY - 1u, j == 0u);
    uint jp1 = select(j + 1u, 0u,      j + 1u == NY);

    uint row    = j   * NX;
    uint row_m  = jm1 * NX;
    uint row_p  = jp1 * NX;

    // Pull streaming.
    // CX = {0, 1, 0,-1, 0, 1,-1,-1, 1}
    // CY = {0, 0, 1, 0,-1, 1, 1,-1,-1}
    float f0 = f_in[0u * N + row   + i  ];
    float f1 = f_in[1u * N + row   + im1];
    float f2 = f_in[2u * N + row_m + i  ];
    float f3 = f_in[3u * N + row   + ip1];
    float f4 = f_in[4u * N + row_p + i  ];
    float f5 = f_in[5u * N + row_m + im1];
    float f6 = f_in[6u * N + row_m + ip1];
    float f7 = f_in[7u * N + row_p + ip1];
    float f8 = f_in[8u * N + row_p + im1];

    // Moments.
    float rho     = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    float inv_rho = 1.0f / rho;
    // Momentum: jx = f1 - f3 + f5 - f6 - f7 + f8
    //           jy = f2 - f4 + f5 + f6 - f7 - f8
    float jx = (f1 - f3) + (f5 + f8) - (f6 + f7);
    float jy = (f2 - f4) + (f5 + f6) - (f7 + f8);
    float ux = jx * inv_rho;
    float uy = jy * inv_rho;

    float usq     = ux * ux + uy * uy;
    float omega   = 1.0f / tau;
    float one_m_w = 1.0f - omega;

    // Pre-scale weights with omega and rho: feq_scaled = omega * W * rho * (...)
    const float W0 = 4.0f / 9.0f;
    const float WS = 1.0f / 9.0f;
    const float WD = 1.0f / 36.0f;

    float orW0 = omega * W0 * rho;
    float orWS = omega * WS * rho;
    float orWD = omega * WD * rho;

    // Common term: A = 1 - 1.5 * usq
    float A = 1.0f - 1.5f * usq;

    // For each k: feq_term = A + 3*cu + 4.5*cu^2
    //           = A + cu * (3 + 4.5*cu)
    // Output = one_m_w * f_k + W_class*omega*rho * feq_term
    uint idx = row + i;

    // k=0: cu = 0 -> feq_term = A
    f_out[0u * N + idx] = fma(one_m_w, f0, orW0 * A);

    // k=1: cu = ux
    {
        float cu = ux;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        f_out[1u * N + idx] = fma(one_m_w, f1, orWS * t);
    }
    // k=2: cu = uy
    {
        float cu = uy;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        f_out[2u * N + idx] = fma(one_m_w, f2, orWS * t);
    }
    // k=3: cu = -ux
    {
        float cu = -ux;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        f_out[3u * N + idx] = fma(one_m_w, f3, orWS * t);
    }
    // k=4: cu = -uy
    {
        float cu = -uy;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        f_out[4u * N + idx] = fma(one_m_w, f4, orWS * t);
    }
    // k=5: cu = ux + uy
    {
        float cu = ux + uy;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        f_out[5u * N + idx] = fma(one_m_w, f5, orWD * t);
    }
    // k=6: cu = -ux + uy
    {
        float cu = uy - ux;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        f_out[6u * N + idx] = fma(one_m_w, f6, orWD * t);
    }
    // k=7: cu = -ux - uy
    {
        float cu = -(ux + uy);
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        f_out[7u * N + idx] = fma(one_m_w, f7, orWD * t);
    }
    // k=8: cu = ux - uy
    {
        float cu = ux - uy;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        f_out[8u * N + idx] = fma(one_m_w, f8, orWD * t);
    }
}
```

Result of previous attempt:
          64x64_50: correct, 0.47 ms, 31.4 GB/s (effective, 72 B/cell) (15.7% of 200 GB/s)
       128x128_100: correct, 0.98 ms, 120.5 GB/s (effective, 72 B/cell) (60.2% of 200 GB/s)
       256x256_100: correct, 2.09 ms, 225.6 GB/s (effective, 72 B/cell) (112.8% of 200 GB/s)
  score (gmean of fraction): 0.4742

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

    uint N  = NX * NY;

    // Precompute wrapped neighbor indices (branchless modulo for ±1).
    uint im1 = (i == 0)      ? (NX - 1) : (i - 1);
    uint ip1 = (i + 1 == NX) ? 0u       : (i + 1);
    uint jm1 = (j == 0)      ? (NY - 1) : (j - 1);
    uint jp1 = (j + 1 == NY) ? 0u       : (j + 1);

    uint row    = j   * NX;
    uint row_m  = jm1 * NX;
    uint row_p  = jp1 * NX;

    // Pull streaming: source = (i - CX[k], j - CY[k]) mod (NX, NY).
    // CX = {0, 1, 0,-1, 0, 1,-1,-1, 1}
    // CY = {0, 0, 1, 0,-1, 1, 1,-1,-1}
    float f0 = f_in[0u * N + row   + i  ];
    float f1 = f_in[1u * N + row   + im1];
    float f2 = f_in[2u * N + row_m + i  ];
    float f3 = f_in[3u * N + row   + ip1];
    float f4 = f_in[4u * N + row_p + i  ];
    float f5 = f_in[5u * N + row_m + im1];
    float f6 = f_in[6u * N + row_m + ip1];
    float f7 = f_in[7u * N + row_p + ip1];
    float f8 = f_in[8u * N + row_p + im1];

    // Moments.
    float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    float inv_rho = 1.0f / rho;
    float ux = ((f1 - f3) + (f5 - f6) - (f7 - f8)) * inv_rho;
    float uy = ((f2 - f4) + (f5 + f6) - (f7 + f8)) * inv_rho;

    float usq     = ux * ux + uy * uy;
    float inv_tau = 1.0f / tau;
    float omega   = inv_tau;
    float one_m_w = 1.0f - omega;

    // Constants.
    const float W0 = 4.0f / 9.0f;
    const float WS = 1.0f / 9.0f;
    const float WD = 1.0f / 36.0f;

    float c1_5_usq = 1.5f * usq;

    // f_eq[k] = W[k] * rho * (1 + 3 cu + 4.5 cu^2 - 1.5 usq)
    // Update: f_out = (1 - omega) * f + omega * feq
    uint idx = row + i;

    // k=0: cu=0
    {
        float feq = W0 * rho * (1.0f - c1_5_usq);
        f_out[0u * N + idx] = one_m_w * f0 + omega * feq;
    }
    // k=1: c=( 1, 0), cu = ux
    {
        float cu = ux;
        float feq = WS * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - c1_5_usq);
        f_out[1u * N + idx] = one_m_w * f1 + omega * feq;
    }
    // k=2: c=( 0, 1), cu = uy
    {
        float cu = uy;
        float feq = WS * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - c1_5_usq);
        f_out[2u * N + idx] = one_m_w * f2 + omega * feq;
    }
    // k=3: c=(-1, 0), cu = -ux
    {
        float cu = -ux;
        float feq = WS * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - c1_5_usq);
        f_out[3u * N + idx] = one_m_w * f3 + omega * feq;
    }
    // k=4: c=( 0,-1), cu = -uy
    {
        float cu = -uy;
        float feq = WS * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - c1_5_usq);
        f_out[4u * N + idx] = one_m_w * f4 + omega * feq;
    }
    // k=5: c=( 1, 1), cu = ux + uy
    {
        float cu = ux + uy;
        float feq = WD * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - c1_5_usq);
        f_out[5u * N + idx] = one_m_w * f5 + omega * feq;
    }
    // k=6: c=(-1, 1), cu = -ux + uy
    {
        float cu = -ux + uy;
        float feq = WD * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - c1_5_usq);
        f_out[6u * N + idx] = one_m_w * f6 + omega * feq;
    }
    // k=7: c=(-1,-1), cu = -ux - uy
    {
        float cu = -ux - uy;
        float feq = WD * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - c1_5_usq);
        f_out[7u * N + idx] = one_m_w * f7 + omega * feq;
    }
    // k=8: c=( 1,-1), cu = ux - uy
    {
        float cu = ux - uy;
        float feq = WD * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - c1_5_usq);
        f_out[8u * N + idx] = one_m_w * f8 + omega * feq;
    }
}
```

Incumbent result:
          64x64_50: correct, 0.47 ms, 31.1 GB/s (effective, 72 B/cell) (15.5% of 200 GB/s)
       128x128_100: correct, 1.00 ms, 118.3 GB/s (effective, 72 B/cell) (59.1% of 200 GB/s)
       256x256_100: correct, 1.82 ms, 259.7 GB/s (effective, 72 B/cell) (129.8% of 200 GB/s)
  score (gmean of fraction): 0.4924

## History

- iter  0: compile=OK | correct=True | score=0.39545824514043965
- iter  1: compile=OK | correct=True | score=0.4924034761153928
- iter  2: compile=OK | correct=True | score=0.4742358593111541

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
