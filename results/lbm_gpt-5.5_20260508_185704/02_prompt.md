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
    const uint i = gid.x;
    const uint j = gid.y;
    if (i >= NX || j >= NY) return;

    const uint N  = NX * NY;
    const uint n1 = N;
    const uint n2 = n1 + N;
    const uint n3 = n2 + N;
    const uint n4 = n3 + N;
    const uint n5 = n4 + N;
    const uint n6 = n5 + N;
    const uint n7 = n6 + N;
    const uint n8 = n7 + N;

    const uint idx = j * NX + i;

    float f0, f1, f2, f3, f4, f5, f6, f7, f8;

    if (i > 0u && i + 1u < NX && j > 0u && j + 1u < NY) {
        f0 = f_in[idx];
        f1 = f_in[n1 + idx - 1u];
        f2 = f_in[n2 + idx - NX];
        f3 = f_in[n3 + idx + 1u];
        f4 = f_in[n4 + idx + NX];
        f5 = f_in[n5 + idx - NX - 1u];
        f6 = f_in[n6 + idx - NX + 1u];
        f7 = f_in[n7 + idx + NX + 1u];
        f8 = f_in[n8 + idx + NX - 1u];
    } else {
        const uint im = (i == 0u) ? (NX - 1u) : (i - 1u);
        const uint ip = (i + 1u == NX) ? 0u : (i + 1u);
        const uint jm = (j == 0u) ? (NY - 1u) : (j - 1u);
        const uint jp = (j + 1u == NY) ? 0u : (j + 1u);

        const uint row  = j  * NX;
        const uint rowm = jm * NX;
        const uint rowp = jp * NX;

        f0 = f_in[row + i];
        f1 = f_in[n1 + row  + im];
        f2 = f_in[n2 + rowm + i ];
        f3 = f_in[n3 + row  + ip];
        f4 = f_in[n4 + rowp + i ];
        f5 = f_in[n5 + rowm + im];
        f6 = f_in[n6 + rowm + ip];
        f7 = f_in[n7 + rowp + ip];
        f8 = f_in[n8 + rowp + im];
    }

    const float rho = ((((((((f0 + f1) + f2) + f3) + f4) + f5) + f6) + f7) + f8);
    const float mx  = (((((f1 - f3) + f5) - f6) - f7) + f8);
    const float my  = (((((f2 - f4) + f5) + f6) - f7) - f8);

    const float inv_rho = 1.0f / rho;
    const float mx2 = mx * mx;
    const float my2 = my * my;
    const float m2  = mx2 + my2;

    const float base_eq = rho - 1.5f * m2 * inv_rho;

    const float eq_x = base_eq + 4.5f * mx2 * inv_rho;
    const float eq_y = base_eq + 4.5f * my2 * inv_rho;

    const float mp = mx + my;
    const float mm = mx - my;
    const float eq_p = base_eq + 4.5f * (mp * mp) * inv_rho;
    const float eq_m = base_eq + 4.5f * (mm * mm) * inv_rho;

    const float omega = 1.0f / tau;
    const float keep  = 1.0f - omega;

    const float ow0 = omega * (4.0f / 9.0f);
    const float ow1 = omega * (1.0f / 9.0f);
    const float ow5 = omega * (1.0f / 36.0f);

    const float tx = 3.0f * mx;
    const float ty = 3.0f * my;
    const float tp = 3.0f * mp;
    const float tm = 3.0f * mm;

    device float *out = f_out + idx;

    out[0 ] = fma(ow0, base_eq,      keep * f0);
    out[n1] = fma(ow1, eq_x + tx,    keep * f1);
    out[n2] = fma(ow1, eq_y + ty,    keep * f2);
    out[n3] = fma(ow1, eq_x - tx,    keep * f3);
    out[n4] = fma(ow1, eq_y - ty,    keep * f4);
    out[n5] = fma(ow5, eq_p + tp,    keep * f5);
    out[n6] = fma(ow5, eq_m - tm,    keep * f6);
    out[n7] = fma(ow5, eq_p - tp,    keep * f7);
    out[n8] = fma(ow5, eq_m + tm,    keep * f8);
}
```

Result of previous attempt:
          64x64_50: correct, 0.49 ms, 30.3 GB/s (effective, 72 B/cell) (15.2% of 200 GB/s)
       128x128_100: correct, 1.99 ms, 59.2 GB/s (effective, 72 B/cell) (29.6% of 200 GB/s)
       256x256_100: correct, 2.31 ms, 204.7 GB/s (effective, 72 B/cell) (102.4% of 200 GB/s)
  score (gmean of fraction): 0.3582

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
          64x64_50: correct, 0.50 ms, 29.7 GB/s (effective, 72 B/cell) (14.8% of 200 GB/s)
       128x128_100: correct, 2.02 ms, 58.5 GB/s (effective, 72 B/cell) (29.2% of 200 GB/s)
       256x256_100: correct, 1.68 ms, 280.3 GB/s (effective, 72 B/cell) (140.2% of 200 GB/s)
  score (gmean of fraction): 0.3932

## History

- iter  0: compile=OK | correct=True | score=0.393238924402302
- iter  1: compile=OK | correct=True | score=0.35817193944057046

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
