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

static inline void lbm_collide_store(device float *f_out,
                                     const uint N,
                                     const uint idx,
                                     const float tau,
                                     const float f0,
                                     const float f1,
                                     const float f2,
                                     const float f3,
                                     const float f4,
                                     const float f5,
                                     const float f6,
                                     const float f7,
                                     const float f8) {
    const float xp = (f1 + f5) + f8;
    const float xn = (f3 + f6) + f7;
    const float rho = ((f0 + f2) + f4) + (xp + xn);
    const float mx = xp - xn;

    const float yp = (f2 + f5) + f6;
    const float yn = (f4 + f7) + f8;
    const float my = yp - yn;

    const float inv_rho = 1.0f / rho;
    const float mx2 = mx * mx;
    const float my2 = my * my;

    const float base = fma(-1.5f, (mx2 + my2) * inv_rho, rho);
    const float q45  = 4.5f * inv_rho;

    const float omega = 1.0f / tau;
    const float keep  = 1.0f - omega;
    const float c0 = omega * (4.0f / 9.0f);
    const float c1 = omega * (1.0f / 9.0f);
    const float c2 = omega * (1.0f / 36.0f);

    device float *pout = f_out;

    pout[idx] = fma(c0, base, keep * f0);

    const float tx = 3.0f * mx;
    const float ty = 3.0f * my;

    const float ex = fma(q45, mx2, base);
    const float ey = fma(q45, my2, base);

    pout += N;
    pout[idx] = fma(c1, ex + tx, keep * f1);

    pout += N;
    pout[idx] = fma(c1, ey + ty, keep * f2);

    pout += N;
    pout[idx] = fma(c1, ex - tx, keep * f3);

    pout += N;
    pout[idx] = fma(c1, ey - ty, keep * f4);

    const float mp = mx + my;
    const float mm = mx - my;

    const float ep = fma(q45, mp * mp, base);
    const float em = fma(q45, mm * mm, base);
    const float tp = tx + ty;
    const float tm = tx - ty;

    pout += N;
    pout[idx] = fma(c2, ep + tp, keep * f5);

    pout += N;
    pout[idx] = fma(c2, em - tm, keep * f6);

    pout += N;
    pout[idx] = fma(c2, ep - tp, keep * f7);

    pout += N;
    pout[idx] = fma(c2, em + tm, keep * f8);
}

static inline void lbm_step_fixed64(device const float *f_in,
                                    device float *f_out,
                                    const float tau,
                                    const uint i,
                                    const uint j) {
    const uint row = j << 6;
    const uint idx = row + i;

    const uint im = (i - 1u) & 63u;
    const uint ip = (i + 1u) & 63u;
    const uint rowm = ((j - 1u) & 63u) << 6;
    const uint rowp = ((j + 1u) & 63u) << 6;

    device const float *pin = f_in;

    const float f0 = pin[idx];
    pin += 4096u;
    const float f1 = pin[row + im];
    pin += 4096u;
    const float f2 = pin[rowm + i];
    pin += 4096u;
    const float f3 = pin[row + ip];
    pin += 4096u;
    const float f4 = pin[rowp + i];
    pin += 4096u;
    const float f5 = pin[rowm + im];
    pin += 4096u;
    const float f6 = pin[rowm + ip];
    pin += 4096u;
    const float f7 = pin[rowp + ip];
    pin += 4096u;
    const float f8 = pin[rowp + im];

    lbm_collide_store(f_out, 4096u, idx, tau, f0, f1, f2, f3, f4, f5, f6, f7, f8);
}

static inline void lbm_step_fixed128(device const float *f_in,
                                     device float *f_out,
                                     const float tau,
                                     const uint i,
                                     const uint j) {
    const uint row = j << 7;
    const uint idx = row + i;

    const uint im = (i - 1u) & 127u;
    const uint ip = (i + 1u) & 127u;
    const uint rowm = ((j - 1u) & 127u) << 7;
    const uint rowp = ((j + 1u) & 127u) << 7;

    device const float *pin = f_in;

    const float f0 = pin[idx];
    pin += 16384u;
    const float f1 = pin[row + im];
    pin += 16384u;
    const float f2 = pin[rowm + i];
    pin += 16384u;
    const float f3 = pin[row + ip];
    pin += 16384u;
    const float f4 = pin[rowp + i];
    pin += 16384u;
    const float f5 = pin[rowm + im];
    pin += 16384u;
    const float f6 = pin[rowm + ip];
    pin += 16384u;
    const float f7 = pin[rowp + ip];
    pin += 16384u;
    const float f8 = pin[rowp + im];

    lbm_collide_store(f_out, 16384u, idx, tau, f0, f1, f2, f3, f4, f5, f6, f7, f8);
}

static inline void lbm_step_fixed256(device const float *f_in,
                                     device float *f_out,
                                     const float tau,
                                     const uint i,
                                     const uint j) {
    const uint row = j << 8;
    const uint idx = row + i;

    float f0, f1, f2, f3, f4, f5, f6, f7, f8;

    if ((i > 0u) && (i < 255u) && (j > 0u) && (j < 255u)) {
        const uint idxm = idx - 256u;
        const uint idxp = idx + 256u;

        device const float *pin = f_in;

        f0 = pin[idx];
        pin += 65536u;
        f1 = pin[idx - 1u];
        pin += 65536u;
        f2 = pin[idxm];
        pin += 65536u;
        f3 = pin[idx + 1u];
        pin += 65536u;
        f4 = pin[idxp];
        pin += 65536u;
        f5 = pin[idxm - 1u];
        pin += 65536u;
        f6 = pin[idxm + 1u];
        pin += 65536u;
        f7 = pin[idxp + 1u];
        pin += 65536u;
        f8 = pin[idxp - 1u];
    } else {
        const uint im = (i - 1u) & 255u;
        const uint ip = (i + 1u) & 255u;
        const uint rowm = ((j - 1u) & 255u) << 8;
        const uint rowp = ((j + 1u) & 255u) << 8;

        device const float *pin = f_in;

        f0 = pin[idx];
        pin += 65536u;
        f1 = pin[row + im];
        pin += 65536u;
        f2 = pin[rowm + i];
        pin += 65536u;
        f3 = pin[row + ip];
        pin += 65536u;
        f4 = pin[rowp + i];
        pin += 65536u;
        f5 = pin[rowm + im];
        pin += 65536u;
        f6 = pin[rowm + ip];
        pin += 65536u;
        f7 = pin[rowp + ip];
        pin += 65536u;
        f8 = pin[rowp + im];
    }

    lbm_collide_store(f_out, 65536u, idx, tau, f0, f1, f2, f3, f4, f5, f6, f7, f8);
}

static inline void lbm_step_general(device const float *f_in,
                                    device float *f_out,
                                    const uint NX,
                                    const uint NY,
                                    const float tau,
                                    const uint i,
                                    const uint j) {
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

    lbm_collide_store(f_out, N, idx, tau, f0, f1, f2, f3, f4, f5, f6, f7, f8);
}

kernel void lbm_step(device const float *f_in   [[buffer(0)]],
                     device       float *f_out  [[buffer(1)]],
                     constant uint        &NX   [[buffer(2)]],
                     constant uint        &NY   [[buffer(3)]],
                     constant float       &tau  [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x;
    const uint j = gid.y;
    if (i >= NX || j >= NY) return;

    if (NX == 64u && NY == 64u) {
        lbm_step_fixed64(f_in, f_out, tau, i, j);
        return;
    }

    if (NX == 128u && NY == 128u) {
        lbm_step_fixed128(f_in, f_out, tau, i, j);
        return;
    }

    if (NX == 256u && NY == 256u) {
        lbm_step_fixed256(f_in, f_out, tau, i, j);
        return;
    }

    lbm_step_general(f_in, f_out, NX, NY, tau, i, j);
}
```

Result of previous attempt:
          64x64_50: correct, 0.29 ms, 50.9 GB/s (effective, 72 B/cell) (25.5% of 200 GB/s)
       128x128_100: correct, 2.08 ms, 56.6 GB/s (effective, 72 B/cell) (28.3% of 200 GB/s)
       256x256_100: correct, 2.33 ms, 202.6 GB/s (effective, 72 B/cell) (101.3% of 200 GB/s)
  score (gmean of fraction): 0.4179

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

- iter  4: compile=OK | correct=True | score=0.3346193348320851
- iter  5: compile=OK | correct=True | score=0.3496501841754343
- iter  6: compile=OK | correct=True | score=0.360218442787931
- iter  7: compile=OK | correct=True | score=0.3515718922758411
- iter  8: compile=OK | correct=True | score=0.3518085650323089
- iter  9: compile=OK | correct=True | score=0.30942080087644286
- iter 10: compile=OK | correct=True | score=0.37110178168958935
- iter 11: compile=OK | correct=True | score=0.4178818116586849

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
