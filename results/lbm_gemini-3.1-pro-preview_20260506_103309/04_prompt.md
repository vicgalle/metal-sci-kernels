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

    uint N = NX * NY;
    uint j_NX = j * NX;
    
    // Fast periodic boundary wrapping, avoiding modulo or branching
    uint ip1 = (i + 1 == NX) ? 0 : i + 1;
    uint im1 = (i == 0) ? NX - 1 : i - 1;
    
    // Y-boundary linear indices evaluated directly
    uint jm1_NX = (j == 0) ? (N - NX) : (j_NX - NX);
    uint jp1_NX = (j + 1 == NY) ? 0 : (j_NX + NX);

    // 1. Pull streaming
    float f0 = f_in[j_NX + i];
    float f1 = f_in[N + j_NX + im1];
    float f2 = f_in[2 * N + jm1_NX + i];
    float f3 = f_in[3 * N + j_NX + ip1];
    float f4 = f_in[4 * N + jp1_NX + i];
    float f5 = f_in[5 * N + jm1_NX + im1];
    float f6 = f_in[6 * N + jm1_NX + ip1];
    float f7 = f_in[7 * N + jp1_NX + ip1];
    float f8 = f_in[8 * N + jp1_NX + im1];

    // 2. Moments computation (tree-reduced for maximal ILP)
    float rho = (f0 + f1 + f2) + (f3 + f4 + f5) + (f6 + f7 + f8);
    float inv_rho = 1.0f / rho;
    
    float ux = ((f1 + f5 + f8) - (f3 + f6 + f7)) * inv_rho;
    float uy = ((f2 + f5 + f6) - (f4 + f7 + f8)) * inv_rho;

    // 3. BGK collision preparation via fully fused Horner polynomials
    float usq = fma(ux, ux, uy * uy); // ux*ux + uy*uy
    float om_usq15 = 1.0f - 1.5f * usq;

    float inv_tau_val = 1.0f / tau;
    float om_inv_tau = 1.0f - inv_tau_val;

    // Pre-bake inverse tau into the equilibrium weights
    float rho_w0_it = rho * (inv_tau_val * (4.0f / 9.0f));
    float rho_w1_it = rho * (inv_tau_val * (1.0f / 9.0f));
    float rho_w5_it = rho * (inv_tau_val * (1.0f / 36.0f));

    // Shared coefficients for Horner polynomial evaluation
    float r1_45 = rho_w1_it * 4.5f;
    float r1_30 = rho_w1_it * 3.0f;
    float r1_om = rho_w1_it * om_usq15;

    float r5_45 = rho_w5_it * 4.5f;
    float r5_30 = rho_w5_it * 3.0f;
    float r5_om = rho_w5_it * om_usq15;

    uint idx = j_NX + i;

    // Evaluate BGK and write out (exactly 3 FMAs per direction)
    f_out[idx]         = fma(f0, om_inv_tau, rho_w0_it * om_usq15);
    f_out[N + idx]     = fma(f1, om_inv_tau, fma( ux, fma( ux, r1_45, r1_30), r1_om));
    f_out[2 * N + idx] = fma(f2, om_inv_tau, fma( uy, fma( uy, r1_45, r1_30), r1_om));
    f_out[3 * N + idx] = fma(f3, om_inv_tau, fma(-ux, fma(-ux, r1_45, r1_30), r1_om));
    f_out[4 * N + idx] = fma(f4, om_inv_tau, fma(-uy, fma(-uy, r1_45, r1_30), r1_om));

    float cu5 = ux + uy;
    f_out[5 * N + idx] = fma(f5, om_inv_tau, fma(cu5, fma(cu5, r5_45, r5_30), r5_om));

    float cu6 = -ux + uy;
    f_out[6 * N + idx] = fma(f6, om_inv_tau, fma(cu6, fma(cu6, r5_45, r5_30), r5_om));

    float cu7 = -cu5;
    f_out[7 * N + idx] = fma(f7, om_inv_tau, fma(cu7, fma(cu7, r5_45, r5_30), r5_om));

    float cu8 = -cu6;
    f_out[8 * N + idx] = fma(f8, om_inv_tau, fma(cu8, fma(cu8, r5_45, r5_30), r5_om));
}
```

Result of previous attempt:
          64x64_50: correct, 0.46 ms, 32.3 GB/s (effective, 72 B/cell) (16.2% of 200 GB/s)
       128x128_100: correct, 1.96 ms, 60.3 GB/s (effective, 72 B/cell) (30.1% of 200 GB/s)
       256x256_100: correct, 2.06 ms, 228.8 GB/s (effective, 72 B/cell) (114.4% of 200 GB/s)
  score (gmean of fraction): 0.3820

## Current best (incumbent)

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

    // Fast periodic boundary wrapping (eliminates slow modulo operations)
    uint ip1 = (i + 1 == NX) ? 0 : i + 1;
    uint im1 = (i == 0) ? NX - 1 : i - 1;
    uint jp1 = (j + 1 == NY) ? 0 : j + 1;
    uint jm1 = (j == 0) ? NY - 1 : j - 1;

    uint N = NX * NY;
    uint j_NX = j * NX;
    uint jm1_NX = jm1 * NX;
    uint jp1_NX = jp1 * NX;

    // 1. Pull streaming
    float f0 = f_in[j_NX + i];
    float f1 = f_in[N + j_NX + im1];
    float f2 = f_in[2 * N + jm1_NX + i];
    float f3 = f_in[3 * N + j_NX + ip1];
    float f4 = f_in[4 * N + jp1_NX + i];
    float f5 = f_in[5 * N + jm1_NX + im1];
    float f6 = f_in[6 * N + jm1_NX + ip1];
    float f7 = f_in[7 * N + jp1_NX + ip1];
    float f8 = f_in[8 * N + jp1_NX + im1];

    // 2. Moments computation
    float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;

    float inv_rho = 1.0f / rho;
    float ux = (f1 - f3 + f5 - f6 - f7 + f8) * inv_rho;
    float uy = (f2 - f4 + f5 + f6 - f7 - f8) * inv_rho;

    // 3. BGK collision preparation (CSE optimizations)
    float usq = ux * ux + uy * uy;
    float usq15 = 1.5f * usq;
    float inv_tau_val = 1.0f / tau;

    float rho_w0 = rho * (4.0f / 9.0f);
    float rho_w1 = rho * (1.0f / 9.0f);
    float rho_w5 = rho * (1.0f / 36.0f);

    uint idx = j_NX + i;

    // Calculate Equilibrium and perfectly match baseline fp-precision order
    float feq0 = rho_w0 * (1.0f - usq15);
    f_out[idx] = f0 - inv_tau_val * (f0 - feq0);

    float cu1 = ux;
    float feq1 = rho_w1 * (1.0f + 3.0f * cu1 + 4.5f * cu1 * cu1 - usq15);
    f_out[N + idx] = f1 - inv_tau_val * (f1 - feq1);

    float cu2 = uy;
    float feq2 = rho_w1 * (1.0f + 3.0f * cu2 + 4.5f * cu2 * cu2 - usq15);
    f_out[2 * N + idx] = f2 - inv_tau_val * (f2 - feq2);

    float cu3 = -ux;
    float feq3 = rho_w1 * (1.0f + 3.0f * cu3 + 4.5f * cu3 * cu3 - usq15);
    f_out[3 * N + idx] = f3 - inv_tau_val * (f3 - feq3);

    float cu4 = -uy;
    float feq4 = rho_w1 * (1.0f + 3.0f * cu4 + 4.5f * cu4 * cu4 - usq15);
    f_out[4 * N + idx] = f4 - inv_tau_val * (f4 - feq4);

    float cu5 = ux + uy;
    float feq5 = rho_w5 * (1.0f + 3.0f * cu5 + 4.5f * cu5 * cu5 - usq15);
    f_out[5 * N + idx] = f5 - inv_tau_val * (f5 - feq5);

    float cu6 = -ux + uy;
    float feq6 = rho_w5 * (1.0f + 3.0f * cu6 + 4.5f * cu6 * cu6 - usq15);
    f_out[6 * N + idx] = f6 - inv_tau_val * (f6 - feq6);

    float cu7 = -ux - uy;
    float feq7 = rho_w5 * (1.0f + 3.0f * cu7 + 4.5f * cu7 * cu7 - usq15);
    f_out[7 * N + idx] = f7 - inv_tau_val * (f7 - feq7);

    float cu8 = ux - uy;
    float feq8 = rho_w5 * (1.0f + 3.0f * cu8 + 4.5f * cu8 * cu8 - usq15);
    f_out[8 * N + idx] = f8 - inv_tau_val * (f8 - feq8);
}
```

Incumbent result:
          64x64_50: correct, 0.47 ms, 31.5 GB/s (effective, 72 B/cell) (15.8% of 200 GB/s)
       128x128_100: correct, 1.70 ms, 69.2 GB/s (effective, 72 B/cell) (34.6% of 200 GB/s)
       256x256_100: correct, 1.92 ms, 245.5 GB/s (effective, 72 B/cell) (122.8% of 200 GB/s)
  score (gmean of fraction): 0.4062

## History

- iter  0: compile=OK | correct=True | score=0.38612320114978643
- iter  1: compile=OK | correct=True | score=0.4061717278202275
- iter  2: compile=OK | correct=True | score=0.3740188819322388
- iter  3: compile=OK | correct=True | score=0.38199258214230564

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
