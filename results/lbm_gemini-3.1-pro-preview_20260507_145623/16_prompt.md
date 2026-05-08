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

    uint N = NX * NY;
    int inx = int(NX);
    int iny = int(NY);

    // Fast 1D periodic boundary offset calculation
    int im1 = (i == 0) ? (inx - 1) : -1;
    int ip1 = (i == NX - 1) ? (1 - inx) : 1;
    int jm1 = (j == 0) ? (inx * (iny - 1)) : -inx;
    int jp1 = (j == NY - 1) ? -(inx * (iny - 1)) : inx;

    // Precalculate base pointers for perfectly uniform strides
    uint i0 = j * NX + i;
    uint i1 = i0 + N;
    uint i2 = i1 + N;
    uint i3 = i2 + N;
    uint i4 = i3 + N;
    uint i5 = i4 + N;
    uint i6 = i5 + N;
    uint i7 = i6 + N;
    uint i8 = i7 + N;

    // Pull streaming reads (relying on 32-bit uint wrapping for negative offsets)
    float f0 = f_in[i0];
    float f1 = f_in[i1 + uint(im1)];
    float f2 = f_in[i2 + uint(jm1)];
    float f3 = f_in[i3 + uint(ip1)];
    float f4 = f_in[i4 + uint(jp1)];
    float f5 = f_in[i5 + uint(im1 + jm1)];
    float f6 = f_in[i6 + uint(ip1 + jm1)];
    float f7 = f_in[i7 + uint(ip1 + jp1)];
    float f8 = f_in[i8 + uint(im1 + jp1)];

    // Moments
    float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    float inv_rho = 1.0f / rho;
    
    float ux = (f1 - f3 + f5 - f6 - f7 + f8) * inv_rho;
    float uy = (f2 - f4 + f5 + f6 - f7 - f8) * inv_rho;

    float ux2 = ux * ux;
    float uy2 = uy * uy;
    float usq = ux2 + uy2;
    float inv_tau = 1.0f / tau;
    float term0 = 1.0f - 1.5f * usq;
    
    float w0_rho = rho * (4.0f / 9.0f);
    float w1_rho = rho * (1.0f / 9.0f);
    float w2_rho = rho * (1.0f / 36.0f);

    // BGK collision and writes (interleaved to minimize register lifespan)
    f_out[i0] = f0 - inv_tau * (f0 - w0_rho * term0);

    float ux3 = 3.0f * ux;
    float ux2_45 = 4.5f * ux2;
    f_out[i1] = f1 - inv_tau * (f1 - w1_rho * (term0 + ux3 + ux2_45));
    f_out[i3] = f3 - inv_tau * (f3 - w1_rho * (term0 - ux3 + ux2_45));

    float uy3 = 3.0f * uy;
    float uy2_45 = 4.5f * uy2;
    f_out[i2] = f2 - inv_tau * (f2 - w1_rho * (term0 + uy3 + uy2_45));
    f_out[i4] = f4 - inv_tau * (f4 - w1_rho * (term0 - uy3 + uy2_45));

    float cu5 = ux + uy;
    float cu5_2 = cu5 * cu5;
    float cu5_3 = 3.0f * cu5;
    float cu5_2_45 = 4.5f * cu5_2;
    f_out[i5] = f5 - inv_tau * (f5 - w2_rho * (term0 + cu5_3 + cu5_2_45));
    f_out[i7] = f7 - inv_tau * (f7 - w2_rho * (term0 - cu5_3 + cu5_2_45));

    float cu6 = -ux + uy;
    float cu6_2 = cu6 * cu6;
    float cu6_3 = 3.0f * cu6;
    float cu6_2_45 = 4.5f * cu6_2;
    f_out[i6] = f6 - inv_tau * (f6 - w2_rho * (term0 + cu6_3 + cu6_2_45));
    f_out[i8] = f8 - inv_tau * (f8 - w2_rho * (term0 - cu6_3 + cu6_2_45));
}
```

Result of previous attempt:
          64x64_50: correct, 0.47 ms, 31.6 GB/s (effective, 72 B/cell) (15.8% of 200 GB/s)
       128x128_100: correct, 1.95 ms, 60.6 GB/s (effective, 72 B/cell) (30.3% of 200 GB/s)
       256x256_100: correct, 2.32 ms, 203.3 GB/s (effective, 72 B/cell) (101.6% of 200 GB/s)
  score (gmean of fraction): 0.3650

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

- iter  8: compile=OK | correct=True | score=0.3497938458825147
- iter  9: compile=FAIL | correct=False | score=N/A
- iter 10: compile=OK | correct=True | score=0.46232862357349447
- iter 11: compile=OK | correct=True | score=0.3909012300348845
- iter 12: compile=OK | correct=True | score=0.3624193042474191
- iter 13: compile=OK | correct=True | score=0.5530764161168673
- iter 14: compile=OK | correct=True | score=0.48881450123883946
- iter 15: compile=OK | correct=True | score=0.36496197947723824

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
