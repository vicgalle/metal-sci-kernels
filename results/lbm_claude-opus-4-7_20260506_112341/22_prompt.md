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

[[max_total_threads_per_threadgroup(256)]]
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

    // Per-plane base pointers.
    device const float *fp0 = f_in;
    device const float *fp1 = fp0 + N;
    device const float *fp2 = fp1 + N;
    device const float *fp3 = fp2 + N;
    device const float *fp4 = fp3 + N;
    device const float *fp5 = fp4 + N;
    device const float *fp6 = fp5 + N;
    device const float *fp7 = fp6 + N;
    device const float *fp8 = fp7 + N;

    // Pull streaming.
    float f0 = fp0[row   + i  ];
    float f1 = fp1[row   + im1];
    float f2 = fp2[row_m + i  ];
    float f3 = fp3[row   + ip1];
    float f4 = fp4[row_p + i  ];
    float f5 = fp5[row_m + im1];
    float f6 = fp6[row_m + ip1];
    float f7 = fp7[row_p + ip1];
    float f8 = fp8[row_p + im1];

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

    const float W0 = 4.0f / 9.0f;
    const float WS = 1.0f / 9.0f;
    const float WD = 1.0f / 36.0f;

    float orW0 = omega * W0 * rho;
    float orWS = omega * WS * rho;
    float orWD = omega * WD * rho;

    // A = 1 - 1.5 * usq
    float A = fma(-1.5f, usq, 1.0f);

    uint idx = row + i;

    // Output plane pointers.
    device float *gp0 = f_out;
    device float *gp1 = gp0 + N;
    device float *gp2 = gp1 + N;
    device float *gp3 = gp2 + N;
    device float *gp4 = gp3 + N;
    device float *gp5 = gp4 + N;
    device float *gp6 = gp5 + N;
    device float *gp7 = gp6 + N;
    device float *gp8 = gp7 + N;

    // k=0
    gp0[idx] = fma(one_m_w, f0, orW0 * A);

    // k=1: cu = ux
    {
        float cu = ux;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        gp1[idx] = fma(one_m_w, f1, orWS * t);
    }
    // k=2: cu = uy
    {
        float cu = uy;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        gp2[idx] = fma(one_m_w, f2, orWS * t);
    }
    // k=3: cu = -ux
    {
        float cu = -ux;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        gp3[idx] = fma(one_m_w, f3, orWS * t);
    }
    // k=4: cu = -uy
    {
        float cu = -uy;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        gp4[idx] = fma(one_m_w, f4, orWS * t);
    }
    // k=5: cu = ux + uy
    {
        float cu = ux + uy;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        gp5[idx] = fma(one_m_w, f5, orWD * t);
    }
    // k=6: cu = uy - ux
    {
        float cu = uy - ux;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        gp6[idx] = fma(one_m_w, f6, orWD * t);
    }
    // k=7: cu = -(ux + uy)
    {
        float cu = -(ux + uy);
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        gp7[idx] = fma(one_m_w, f7, orWD * t);
    }
    // k=8: cu = ux - uy
    {
        float cu = ux - uy;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        gp8[idx] = fma(one_m_w, f8, orWD * t);
    }
}
```

Result of previous attempt:
          64x64_50: correct, 0.26 ms, 57.5 GB/s (effective, 72 B/cell) (28.8% of 200 GB/s)
       128x128_100: correct, 1.20 ms, 98.2 GB/s (effective, 72 B/cell) (49.1% of 200 GB/s)
       256x256_100: correct, 2.16 ms, 218.8 GB/s (effective, 72 B/cell) (109.4% of 200 GB/s)
  score (gmean of fraction): 0.5366

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

- iter 14: compile=OK | correct=True | score=0.3563681761227277
- iter 15: compile=OK | correct=True | score=0.37156146573215193
- iter 16: compile=OK | correct=True | score=0.3736734976130389
- iter 17: compile=OK | correct=True | score=0.38803881714745053
- iter 18: compile=OK | correct=True | score=0.38634373724939486
- iter 19: compile=OK | correct=True | score=0.3967944229044034
- iter 20: compile=OK | correct=True | score=0.3813302318042059
- iter 21: compile=OK | correct=True | score=0.5366310976649512

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
