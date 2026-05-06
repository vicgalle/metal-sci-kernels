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

[[max_total_threads_per_threadgroup(32)]]
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

    // Pull-streaming addresses.
    uint a0 = row   + i;
    uint a1 = row   + im1;
    uint a2 = row_m + i;
    uint a3 = row   + ip1;
    uint a4 = row_p + i;
    uint a5 = row_m + im1;
    uint a6 = row_m + ip1;
    uint a7 = row_p + ip1;
    uint a8 = row_p + im1;

    // Plane base pointers — independent address generation.
    device const float *p0 = f_in;
    device const float *p1 = p0 + N;
    device const float *p2 = p1 + N;
    device const float *p3 = p2 + N;
    device const float *p4 = p3 + N;
    device const float *p5 = p4 + N;
    device const float *p6 = p5 + N;
    device const float *p7 = p6 + N;
    device const float *p8 = p7 + N;

    float f0 = p0[a0];
    float f1 = p1[a1];
    float f2 = p2[a2];
    float f3 = p3[a3];
    float f4 = p4[a4];
    float f5 = p5[a5];
    float f6 = p6[a6];
    float f7 = p7[a7];
    float f8 = p8[a8];

    // Moments.
    float rho     = ((f0 + f1) + (f2 + f3)) + ((f4 + f5) + (f6 + f7)) + f8;
    float inv_rho = 1.0f / rho;
    float jx = (f1 - f3) + (f5 + f8) - (f6 + f7);
    float jy = (f2 - f4) + (f5 + f6) - (f7 + f8);
    float ux = jx * inv_rho;
    float uy = jy * inv_rho;

    float usq     = fma(ux, ux, uy * uy);
    float omega   = 1.0f / tau;
    float one_m_w = 1.0f - omega;

    const float W0 = 4.0f / 9.0f;
    const float WS = 1.0f / 9.0f;
    const float WD = 1.0f / 36.0f;

    float or_rho = omega * rho;
    float orW0   = or_rho * W0;
    float orWS   = or_rho * WS;
    float orWD   = or_rho * WD;

    float A = fma(-1.5f, usq, 1.0f);

    uint idx = a0;

    device float *q0 = f_out;
    device float *q1 = q0 + N;
    device float *q2 = q1 + N;
    device float *q3 = q2 + N;
    device float *q4 = q3 + N;
    device float *q5 = q4 + N;
    device float *q6 = q5 + N;
    device float *q7 = q6 + N;
    device float *q8 = q7 + N;

    // k=0
    q0[idx] = fma(one_m_w, f0, orW0 * A);

    {
        float cu = ux;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        q1[idx] = fma(one_m_w, f1, orWS * t);
    }
    {
        float cu = uy;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        q2[idx] = fma(one_m_w, f2, orWS * t);
    }
    {
        float cu = -ux;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        q3[idx] = fma(one_m_w, f3, orWS * t);
    }
    {
        float cu = -uy;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        q4[idx] = fma(one_m_w, f4, orWS * t);
    }
    {
        float cu = ux + uy;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        q5[idx] = fma(one_m_w, f5, orWD * t);
    }
    {
        float cu = uy - ux;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        q6[idx] = fma(one_m_w, f6, orWD * t);
    }
    {
        float cu = -(ux + uy);
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        q7[idx] = fma(one_m_w, f7, orWD * t);
    }
    {
        float cu = ux - uy;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        q8[idx] = fma(one_m_w, f8, orWD * t);
    }
}
```

Result of previous attempt:
          64x64_50: correct, 0.52 ms, 28.4 GB/s (effective, 72 B/cell) (14.2% of 200 GB/s)
       128x128_100: correct, 2.20 ms, 53.6 GB/s (effective, 72 B/cell) (26.8% of 200 GB/s)
       256x256_100: correct, 1.98 ms, 238.2 GB/s (effective, 72 B/cell) (119.1% of 200 GB/s)
  score (gmean of fraction): 0.3564

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(64)]]
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
    float jx = (f1 - f3) + (f5 + f8) - (f6 + f7);
    float jy = (f2 - f4) + (f5 + f6) - (f7 + f8);
    float ux = jx * inv_rho;
    float uy = jy * inv_rho;

    float usq     = ux * ux + uy * uy;
    float omega   = 1.0f / tau;
    float one_m_w = 1.0f - omega;

    // Pre-scale: omega * W[k] * rho
    const float W0 = 4.0f / 9.0f;
    const float WS = 1.0f / 9.0f;
    const float WD = 1.0f / 36.0f;

    float orW0 = omega * W0 * rho;
    float orWS = omega * WS * rho;
    float orWD = omega * WD * rho;

    // A = 1 - 1.5 * usq
    float A = fma(-1.5f, usq, 1.0f);

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
    // k=7: cu = -(ux + uy)
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

Incumbent result:
          64x64_50: correct, 0.23 ms, 63.3 GB/s (effective, 72 B/cell) (31.6% of 200 GB/s)
       128x128_100: correct, 1.22 ms, 96.6 GB/s (effective, 72 B/cell) (48.3% of 200 GB/s)
       256x256_100: correct, 2.27 ms, 207.4 GB/s (effective, 72 B/cell) (103.7% of 200 GB/s)
  score (gmean of fraction): 0.5412

## History

- iter  7: compile=OK | correct=True | score=0.35738053716292467
- iter  8: compile=OK | correct=True | score=0.3940782285414889
- iter  9: compile=OK | correct=True | score=0.369744520322877
- iter 10: compile=OK | correct=True | score=0.3794652074667974
- iter 11: compile=OK | correct=True | score=0.37421078578654315
- iter 12: compile=OK | correct=True | score=0.39268414713881794
- iter 13: compile=OK | correct=True | score=0.3562957819366984
- iter 14: compile=OK | correct=True | score=0.3563681761227277

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
