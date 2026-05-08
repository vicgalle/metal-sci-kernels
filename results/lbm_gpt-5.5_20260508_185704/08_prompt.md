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

    if (NX == 256u && NY == 256u) {
        const uint N    = 65536u;
        const uint row  = j << 8;
        const uint idx  = row | i;
        const uint im   = (i - 1u) & 255u;
        const uint ip   = (i + 1u) & 255u;
        const uint rowm = ((j - 1u) & 255u) << 8;
        const uint rowp = ((j + 1u) & 255u) << 8;

        const float f0 = f_in[idx];
        const float f1 = f_in[N       + (row  | im)];
        const float f2 = f_in[N * 2u  + (rowm | i )];
        const float f3 = f_in[N * 3u  + (row  | ip)];
        const float f4 = f_in[N * 4u  + (rowp | i )];
        const float f5 = f_in[N * 5u  + (rowm | im)];
        const float f6 = f_in[N * 6u  + (rowm | ip)];
        const float f7 = f_in[N * 7u  + (rowp | ip)];
        const float f8 = f_in[N * 8u  + (rowp | im)];

        const float s13 = f1 + f3;
        const float s24 = f2 + f4;
        const float s57 = f5 + f7;
        const float s68 = f6 + f8;
        const float rho = ((f0 + s13) + (s24 + s57)) + s68;

        const float d13 = f1 - f3;
        const float d24 = f2 - f4;
        const float d57 = f5 - f7;
        const float d68 = f6 - f8;
        const float mx = (d13 + d57) - d68;
        const float my = (d24 + d57) + d68;

        const float inv_rho = 1.0f / rho;
        const float mx2 = mx * mx;
        const float my2 = my * my;
        const float m2  = mx2 + my2;
        const float m2r = m2 * inv_rho;

        const float base = fma(-1.5f, m2r, rho);
        const float q45  = 4.5f * inv_rho;

        const float tx = 3.0f * mx;
        const float ty = 3.0f * my;

        const float ex = fma(q45, mx2, base);
        const float ey = fma(q45, my2, base);

        const float diag0 = fma(3.0f, m2r, rho);
        const float cross = (9.0f * inv_rho) * (mx * my);
        const float ep = diag0 + cross;
        const float em = diag0 - cross;

        const float tp = tx + ty;
        const float tm = tx - ty;

        const float omega = 1.0f / tau;
        const float om    = 1.0f - omega;
        const float ow1   = omega * (1.0f / 9.0f);
        const float ow0   = 4.0f * ow1;
        const float owd   = 0.25f * ow1;

        f_out[idx]              = fma(ow0, base,    om * f0);
        f_out[N       + idx]    = fma(ow1, ex + tx, om * f1);
        f_out[N * 2u  + idx]    = fma(ow1, ey + ty, om * f2);
        f_out[N * 3u  + idx]    = fma(ow1, ex - tx, om * f3);
        f_out[N * 4u  + idx]    = fma(ow1, ey - ty, om * f4);
        f_out[N * 5u  + idx]    = fma(owd, ep + tp, om * f5);
        f_out[N * 6u  + idx]    = fma(owd, em - tm, om * f6);
        f_out[N * 7u  + idx]    = fma(owd, ep - tp, om * f7);
        f_out[N * 8u  + idx]    = fma(owd, em + tm, om * f8);
        return;
    }

    const uint N   = NX * NY;
    const uint row = j * NX;
    const uint idx = row + i;

    const uint im   = (i == 0u) ? (NX - 1u) : (i - 1u);
    const uint ip   = (i + 1u == NX) ? 0u : (i + 1u);
    const uint rowm = (j == 0u) ? (N - NX) : (row - NX);
    const uint rowp = (j + 1u == NY) ? 0u : (row + NX);

    device const float *pin = f_in;

    const float f0 = pin[idx];
    pin += N;
    const float f1 = pin[row + im];
    pin += N;
    const float f2 = pin[rowm + i];
    pin += N;
    const float f3 = pin[row + ip];
    pin += N;
    const float f4 = pin[rowp + i];
    pin += N;
    const float f5 = pin[rowm + im];
    pin += N;
    const float f6 = pin[rowm + ip];
    pin += N;
    const float f7 = pin[rowp + ip];
    pin += N;
    const float f8 = pin[rowp + im];

    const float rho = (((f0 + f1) + (f2 + f3)) +
                       ((f4 + f5) + (f6 + f7))) + f8;

    const float mx = (((f1 - f3) + f5) - f6) - f7 + f8;
    const float my = (((f2 - f4) + f5) + f6) - f7 - f8;

    const float inv_rho = 1.0f / rho;
    const float mx2 = mx * mx;
    const float my2 = my * my;

    const float base = fma(-1.5f, (mx2 + my2) * inv_rho, rho);
    const float q45  = 4.5f * inv_rho;

    const float omega = 1.0f / tau;

    device float *pout = f_out;

    pout[idx] = fma(omega, (4.0f / 9.0f) * base - f0, f0);

    const float tx = 3.0f * mx;
    const float ty = 3.0f * my;

    const float ex = fma(q45, mx2, base);
    const float ey = fma(q45, my2, base);

    pout += N;
    pout[idx] = fma(omega, (1.0f / 9.0f) * (ex + tx) - f1, f1);

    pout += N;
    pout[idx] = fma(omega, (1.0f / 9.0f) * (ey + ty) - f2, f2);

    pout += N;
    pout[idx] = fma(omega, (1.0f / 9.0f) * (ex - tx) - f3, f3);

    pout += N;
    pout[idx] = fma(omega, (1.0f / 9.0f) * (ey - ty) - f4, f4);

    const float mp = mx + my;
    const float mm = mx - my;
    const float mp2 = mp * mp;
    const float mm2 = mm * mm;

    const float ep = fma(q45, mp2, base);
    const float em = fma(q45, mm2, base);
    const float tp = 3.0f * mp;
    const float tm = 3.0f * mm;

    pout += N;
    pout[idx] = fma(omega, (1.0f / 36.0f) * (ep + tp) - f5, f5);

    pout += N;
    pout[idx] = fma(omega, (1.0f / 36.0f) * (em - tm) - f6, f6);

    pout += N;
    pout[idx] = fma(omega, (1.0f / 36.0f) * (ep - tp) - f7, f7);

    pout += N;
    pout[idx] = fma(omega, (1.0f / 36.0f) * (em + tm) - f8, f8);
}
```

Result of previous attempt:
          64x64_50: correct, 0.50 ms, 29.5 GB/s (effective, 72 B/cell) (14.7% of 200 GB/s)
       128x128_100: correct, 2.06 ms, 57.2 GB/s (effective, 72 B/cell) (28.6% of 200 GB/s)
       256x256_100: correct, 2.29 ms, 206.3 GB/s (effective, 72 B/cell) (103.1% of 200 GB/s)
  score (gmean of fraction): 0.3516

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
    const uint i = gid.x;
    const uint j = gid.y;
    if (i >= NX || j >= NY) return;

    const uint N   = NX * NY;
    const uint row = j * NX;
    const uint idx = row + i;

    const uint im   = (i == 0u) ? (NX - 1u) : (i - 1u);
    const uint ip   = (i + 1u == NX) ? 0u : (i + 1u);
    const uint rowm = (j == 0u) ? (N - NX) : (row - NX);
    const uint rowp = (j + 1u == NY) ? 0u : (row + NX);

    device const float *pin = f_in;

    const float f0 = pin[idx];
    pin += N;
    const float f1 = pin[row + im];
    pin += N;
    const float f2 = pin[rowm + i];
    pin += N;
    const float f3 = pin[row + ip];
    pin += N;
    const float f4 = pin[rowp + i];
    pin += N;
    const float f5 = pin[rowm + im];
    pin += N;
    const float f6 = pin[rowm + ip];
    pin += N;
    const float f7 = pin[rowp + ip];
    pin += N;
    const float f8 = pin[rowp + im];

    const float rho = (((f0 + f1) + (f2 + f3)) +
                       ((f4 + f5) + (f6 + f7))) + f8;

    const float mx = (((f1 - f3) + f5) - f6) - f7 + f8;
    const float my = (((f2 - f4) + f5) + f6) - f7 - f8;

    const float inv_rho = 1.0f / rho;
    const float mx2 = mx * mx;
    const float my2 = my * my;

    const float base = fma(-1.5f, (mx2 + my2) * inv_rho, rho);
    const float q45  = 4.5f * inv_rho;

    const float omega = 1.0f / tau;

    device float *pout = f_out;

    pout[idx] = fma(omega, (4.0f / 9.0f) * base - f0, f0);

    const float tx = 3.0f * mx;
    const float ty = 3.0f * my;

    const float ex = fma(q45, mx2, base);
    const float ey = fma(q45, my2, base);

    pout += N;
    pout[idx] = fma(omega, (1.0f / 9.0f) * (ex + tx) - f1, f1);

    pout += N;
    pout[idx] = fma(omega, (1.0f / 9.0f) * (ey + ty) - f2, f2);

    pout += N;
    pout[idx] = fma(omega, (1.0f / 9.0f) * (ex - tx) - f3, f3);

    pout += N;
    pout[idx] = fma(omega, (1.0f / 9.0f) * (ey - ty) - f4, f4);

    const float mp = mx + my;
    const float mm = mx - my;
    const float mp2 = mp * mp;
    const float mm2 = mm * mm;

    const float ep = fma(q45, mp2, base);
    const float em = fma(q45, mm2, base);
    const float tp = 3.0f * mp;
    const float tm = 3.0f * mm;

    pout += N;
    pout[idx] = fma(omega, (1.0f / 36.0f) * (ep + tp) - f5, f5);

    pout += N;
    pout[idx] = fma(omega, (1.0f / 36.0f) * (em - tm) - f6, f6);

    pout += N;
    pout[idx] = fma(omega, (1.0f / 36.0f) * (ep - tp) - f7, f7);

    pout += N;
    pout[idx] = fma(omega, (1.0f / 36.0f) * (em + tm) - f8, f8);
}
```

Incumbent result:
          64x64_50: correct, 0.22 ms, 68.1 GB/s (effective, 72 B/cell) (34.1% of 200 GB/s)
       128x128_100: correct, 1.19 ms, 99.0 GB/s (effective, 72 B/cell) (49.5% of 200 GB/s)
       256x256_100: correct, 2.86 ms, 164.8 GB/s (effective, 72 B/cell) (82.4% of 200 GB/s)
  score (gmean of fraction): 0.5179

## History

- iter  0: compile=OK | correct=True | score=0.393238924402302
- iter  1: compile=OK | correct=True | score=0.35817193944057046
- iter  2: compile=OK | correct=True | score=0.5178946510465539
- iter  3: compile=OK | correct=True | score=0.34537143965206346
- iter  4: compile=OK | correct=True | score=0.3346193348320851
- iter  5: compile=OK | correct=True | score=0.3496501841754343
- iter  6: compile=OK | correct=True | score=0.360218442787931
- iter  7: compile=OK | correct=True | score=0.3515718922758411

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
