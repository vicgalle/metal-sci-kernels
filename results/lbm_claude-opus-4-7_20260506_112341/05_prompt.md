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

    // Branchless periodic neighbors.
    uint im1 = (i == 0u)        ? (NX - 1u) : (i - 1u);
    uint ip1 = (i + 1u == NX)   ? 0u        : (i + 1u);
    uint jm1 = (j == 0u)        ? (NY - 1u) : (j - 1u);
    uint jp1 = (j + 1u == NY)   ? 0u        : (j + 1u);

    uint row    = j   * NX;
    uint row_m  = jm1 * NX;
    uint row_p  = jp1 * NX;

    // Issue all 9 loads first to expose maximum memory-level parallelism.
    // CX = {0, 1, 0,-1, 0, 1,-1,-1, 1}
    // CY = {0, 0, 1, 0,-1, 1, 1,-1,-1}
    device const float *p0 = f_in + 0u * N;
    device const float *p1 = f_in + 1u * N;
    device const float *p2 = f_in + 2u * N;
    device const float *p3 = f_in + 3u * N;
    device const float *p4 = f_in + 4u * N;
    device const float *p5 = f_in + 5u * N;
    device const float *p6 = f_in + 6u * N;
    device const float *p7 = f_in + 7u * N;
    device const float *p8 = f_in + 8u * N;

    float f0 = p0[row   + i  ];
    float f1 = p1[row   + im1];
    float f2 = p2[row_m + i  ];
    float f3 = p3[row   + ip1];
    float f4 = p4[row_p + i  ];
    float f5 = p5[row_m + im1];
    float f6 = p6[row_m + ip1];
    float f7 = p7[row_p + ip1];
    float f8 = p8[row_p + im1];

    // Moments.
    float rho     = ((f0 + f1) + (f2 + f3)) + ((f4 + f5) + (f6 + f7)) + f8;
    float inv_rho = 1.0f / rho;
    float jx = (f1 - f3) + (f5 - f6) + (f8 - f7);
    float jy = (f2 - f4) + (f5 + f6) - (f7 + f8);
    float ux = jx * inv_rho;
    float uy = jy * inv_rho;

    float usq     = fma(ux, ux, uy * uy);
    float omega   = 1.0f / tau;
    float one_m_w = 1.0f - omega;

    const float W0 = 4.0f / 9.0f;
    const float WS = 1.0f / 9.0f;
    const float WD = 1.0f / 36.0f;

    float orho  = omega * rho;
    float orW0  = orho * W0;
    float orWS  = orho * WS;
    float orWD  = orho * WD;

    // A = 1 - 1.5 * usq
    float A = fma(-1.5f, usq, 1.0f);

    uint idx = row + i;

    // For each k: t = A + cu * (3 + 4.5 * cu)
    // f_out = (1 - omega) * f_k + orW * t

    float cu1 = ux;
    float cu2 = uy;
    float cu5 = ux + uy;
    float cu6 = uy - ux;

    float t1 = A + cu1 * fma(4.5f, cu1, 3.0f);
    float t2 = A + cu2 * fma(4.5f, cu2, 3.0f);
    float t3 = A + (-cu1) * fma(4.5f, -cu1, 3.0f);
    float t4 = A + (-cu2) * fma(4.5f, -cu2, 3.0f);
    float t5 = A + cu5 * fma(4.5f, cu5, 3.0f);
    float t6 = A + cu6 * fma(4.5f, cu6, 3.0f);
    float t7 = A + (-cu5) * fma(4.5f, -cu5, 3.0f);
    float t8 = A + (-cu6) * fma(4.5f, -cu6, 3.0f);

    f_out[0u * N + idx] = fma(one_m_w, f0, orW0 * A);
    f_out[1u * N + idx] = fma(one_m_w, f1, orWS * t1);
    f_out[2u * N + idx] = fma(one_m_w, f2, orWS * t2);
    f_out[3u * N + idx] = fma(one_m_w, f3, orWS * t3);
    f_out[4u * N + idx] = fma(one_m_w, f4, orWS * t4);
    f_out[5u * N + idx] = fma(one_m_w, f5, orWD * t5);
    f_out[6u * N + idx] = fma(one_m_w, f6, orWD * t6);
    f_out[7u * N + idx] = fma(one_m_w, f7, orWD * t7);
    f_out[8u * N + idx] = fma(one_m_w, f8, orWD * t8);
}
```

Result of previous attempt:
          64x64_50: correct, 0.49 ms, 30.0 GB/s (effective, 72 B/cell) (15.0% of 200 GB/s)
       128x128_100: correct, 2.13 ms, 55.5 GB/s (effective, 72 B/cell) (27.8% of 200 GB/s)
       256x256_100: correct, 2.04 ms, 231.2 GB/s (effective, 72 B/cell) (115.6% of 200 GB/s)
  score (gmean of fraction): 0.3637

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

- iter  0: compile=OK | correct=True | score=0.39545824514043965
- iter  1: compile=OK | correct=True | score=0.4924034761153928
- iter  2: compile=OK | correct=True | score=0.4742358593111541
- iter  3: compile=OK | correct=True | score=0.5411587265744036
- iter  4: compile=OK | correct=True | score=0.3637133364013231

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
