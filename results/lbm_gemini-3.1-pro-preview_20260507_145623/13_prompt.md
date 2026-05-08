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
    const uint nx = NX;
    const uint ny = NY;
    
    if (gid.x >= nx || gid.y >= ny) return;

    const uint i = gid.x;
    const uint j = gid.y;
    const uint N = nx * ny;

    // Fast periodic boundary wrapping using conditionally selected constants
    const uint im1 = (i > 0) ? i - 1 : nx - 1;
    const uint ip1 = (i == nx - 1) ? 0 : i + 1;
    const uint jm1 = (j > 0) ? j - 1 : ny - 1;
    const uint jp1 = (j == ny - 1) ? 0 : j + 1;

    const uint y_j   = j * nx;
    const uint y_jm1 = jm1 * nx;
    const uint y_jp1 = jp1 * nx;

    const uint idx = y_j + i;

    // Fully scalarized pull-streaming memory fetches map directly to registers
    // and naturally benefit from adjacent 32-thread coalesced vector loads.
    float f0 = f_in[idx];
    float f1 = f_in[1 * N + y_j + im1];
    float f2 = f_in[2 * N + y_jm1 + i];
    float f3 = f_in[3 * N + y_j + ip1];
    float f4 = f_in[4 * N + y_jp1 + i];
    float f5 = f_in[5 * N + y_jm1 + im1];
    float f6 = f_in[6 * N + y_jm1 + ip1];
    float f7 = f_in[7 * N + y_jp1 + ip1];
    float f8 = f_in[8 * N + y_jp1 + im1];

    // Compute macroscopic moments
    float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    float inv_rho = 1.0f / rho;
    
    float ux = (f1 - f3 + f5 - f6 - f7 + f8) * inv_rho;
    float uy = (f2 - f4 + f5 + f6 - f7 - f8) * inv_rho;

    // Precompute constants and base equilibrium terms
    float usq_term = 1.0f - 1.5f * (ux * ux + uy * uy);
    
    float omega = 1.0f / tau;
    float om_omega = 1.0f - omega;
    float omega_rho = omega * rho;

    // Distribute omega into the polynomial coefficients early
    float w0_rho = omega_rho * (4.0f / 9.0f);
    float w1_rho = omega_rho * (1.0f / 9.0f);
    float w5_rho = omega_rho * (1.0f / 36.0f);

    float w0_usq = w0_rho * usq_term;
    float w1_usq = w1_rho * usq_term;
    float w5_usq = w5_rho * usq_term;

    float w1_rho_3  = w1_rho * 3.0f;
    float w1_rho_45 = w1_rho * 4.5f;
    float w5_rho_3  = w5_rho * 3.0f;
    float w5_rho_45 = w5_rho * 4.5f;

    // Axis directions (W[1..4])
    // Opposite directions perfectly share the quadratic component.
    float ux_sq = ux * ux;
    float uy_sq = uy * uy;
    
    float t1_base = fma(w1_rho_45, ux_sq, w1_usq);
    float t2_base = fma(w1_rho_45, uy_sq, w1_usq);

    float w1_ux_3 = ux * w1_rho_3;
    float w1_uy_3 = uy * w1_rho_3;

    float feq1 = t1_base + w1_ux_3;
    float feq3 = t1_base - w1_ux_3;
    float feq2 = t2_base + w1_uy_3;
    float feq4 = t2_base - w1_uy_3;

    // Diagonal directions (W[5..8])
    float cu5 = ux + uy;
    float cu8 = ux - uy;
    
    float cu5_sq = cu5 * cu5;
    float cu8_sq = cu8 * cu8;
    
    float t5_base = fma(w5_rho_45, cu5_sq, w5_usq);
    float t8_base = fma(w5_rho_45, cu8_sq, w5_usq);
    
    float w5_cu5_3 = cu5 * w5_rho_3;
    float w5_cu8_3 = cu8 * w5_rho_3;
    
    // Note: cu6 = -cu8 and cu7 = -cu5, so t8_base handles feq6, and t5_base handles feq7.
    float feq5 = t5_base + w5_cu5_3;
    float feq7 = t5_base - w5_cu5_3;
    float feq8 = t8_base + w5_cu8_3;
    float feq6 = t8_base - w5_cu8_3;

    // Fused BGK scatter: f_out = f_in * (1 - omega) + omega_feq
    f_out[idx]         = fma(f0, om_omega, w0_usq);
    f_out[1 * N + idx] = fma(f1, om_omega, feq1);
    f_out[2 * N + idx] = fma(f2, om_omega, feq2);
    f_out[3 * N + idx] = fma(f3, om_omega, feq3);
    f_out[4 * N + idx] = fma(f4, om_omega, feq4);
    f_out[5 * N + idx] = fma(f5, om_omega, feq5);
    f_out[6 * N + idx] = fma(f6, om_omega, feq6);
    f_out[7 * N + idx] = fma(f7, om_omega, feq7);
    f_out[8 * N + idx] = fma(f8, om_omega, feq8);
}
```

Result of previous attempt:
          64x64_50: correct, 0.47 ms, 31.5 GB/s (effective, 72 B/cell) (15.8% of 200 GB/s)
       128x128_100: correct, 2.02 ms, 58.3 GB/s (effective, 72 B/cell) (29.1% of 200 GB/s)
       256x256_100: correct, 2.28 ms, 207.4 GB/s (effective, 72 B/cell) (103.7% of 200 GB/s)
  score (gmean of fraction): 0.3624

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

- iter  5: compile=OK | correct=True | score=0.35137941309385284
- iter  6: compile=OK | correct=True | score=0.4429690944125417
- iter  7: compile=OK | correct=True | score=0.43205974293706484
- iter  8: compile=OK | correct=True | score=0.3497938458825147
- iter  9: compile=FAIL | correct=False | score=N/A
- iter 10: compile=OK | correct=True | score=0.46232862357349447
- iter 11: compile=OK | correct=True | score=0.3909012300348845
- iter 12: compile=OK | correct=True | score=0.3624193042474191

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
