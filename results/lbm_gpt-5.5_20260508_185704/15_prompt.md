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

#define LBM_COLLIDE_STORE_OMEGA(FOUT_, N_, IDX_, OMEGA_, F0_, F1_, F2_, F3_, F4_, F5_, F6_, F7_, F8_) do { \
    const float rho = ((((F0_) + (F1_)) + ((F2_) + (F3_))) + (((F4_) + (F5_)) + ((F6_) + (F7_)))) + (F8_); \
    const float mx = (((((F1_) - (F3_)) + (F5_)) - (F6_)) - (F7_)) + (F8_); \
    const float my = (((((F2_) - (F4_)) + (F5_)) + (F6_)) - (F7_)) - (F8_); \
    const float inv_rho = 1.0f / rho; \
    const float mx2 = mx * mx; \
    const float my2 = my * my; \
    const float base = fma(-1.5f, (mx2 + my2) * inv_rho, rho); \
    const float q45 = 4.5f * inv_rho; \
    const float omega_lbm = (OMEGA_); \
    device float *pout = (FOUT_); \
    pout[(IDX_)] = fma(omega_lbm, (4.0f / 9.0f) * base - (F0_), (F0_)); \
    const float tx = 3.0f * mx; \
    const float ty = 3.0f * my; \
    const float ex = fma(q45, mx2, base); \
    const float ey = fma(q45, my2, base); \
    pout += (N_); \
    pout[(IDX_)] = fma(omega_lbm, (1.0f / 9.0f) * (ex + tx) - (F1_), (F1_)); \
    pout += (N_); \
    pout[(IDX_)] = fma(omega_lbm, (1.0f / 9.0f) * (ey + ty) - (F2_), (F2_)); \
    pout += (N_); \
    pout[(IDX_)] = fma(omega_lbm, (1.0f / 9.0f) * (ex - tx) - (F3_), (F3_)); \
    pout += (N_); \
    pout[(IDX_)] = fma(omega_lbm, (1.0f / 9.0f) * (ey - ty) - (F4_), (F4_)); \
    const float mp = mx + my; \
    const float mm = mx - my; \
    const float mp2 = mp * mp; \
    const float mm2 = mm * mm; \
    const float ep = fma(q45, mp2, base); \
    const float em = fma(q45, mm2, base); \
    const float tp = 3.0f * mp; \
    const float tm = 3.0f * mm; \
    pout += (N_); \
    pout[(IDX_)] = fma(omega_lbm, (1.0f / 36.0f) * (ep + tp) - (F5_), (F5_)); \
    pout += (N_); \
    pout[(IDX_)] = fma(omega_lbm, (1.0f / 36.0f) * (em - tm) - (F6_), (F6_)); \
    pout += (N_); \
    pout[(IDX_)] = fma(omega_lbm, (1.0f / 36.0f) * (ep - tp) - (F7_), (F7_)); \
    pout += (N_); \
    pout[(IDX_)] = fma(omega_lbm, (1.0f / 36.0f) * (em + tm) - (F8_), (F8_)); \
} while (0)

#define LBM_POWER2_FULL_CASE(SZ_, SHIFT_, NVAL_) \
    if (NX == (SZ_) && NY == (SZ_)) { \
        const uint row = j << (SHIFT_); \
        const uint idx = row + i; \
        float f0, f1, f2, f3, f4, f5, f6, f7, f8; \
        if ((i - 1u < ((SZ_) - 2u)) && (j - 1u < ((SZ_) - 2u))) { \
            const uint idxm = idx - (SZ_); \
            const uint idxp = idx + (SZ_); \
            device const float *pin = f_in; \
            f0 = pin[idx]; \
            pin += (NVAL_); \
            f1 = pin[idx - 1u]; \
            pin += (NVAL_); \
            f2 = pin[idxm]; \
            pin += (NVAL_); \
            f3 = pin[idx + 1u]; \
            pin += (NVAL_); \
            f4 = pin[idxp]; \
            pin += (NVAL_); \
            f5 = pin[idxm - 1u]; \
            pin += (NVAL_); \
            f6 = pin[idxm + 1u]; \
            pin += (NVAL_); \
            f7 = pin[idxp + 1u]; \
            pin += (NVAL_); \
            f8 = pin[idxp - 1u]; \
        } else { \
            const uint mask = (SZ_) - 1u; \
            const uint im = (i - 1u) & mask; \
            const uint ip = (i + 1u) & mask; \
            const uint rowm = ((j - 1u) & mask) << (SHIFT_); \
            const uint rowp = ((j + 1u) & mask) << (SHIFT_); \
            device const float *pin = f_in; \
            f0 = pin[idx]; \
            pin += (NVAL_); \
            f1 = pin[row + im]; \
            pin += (NVAL_); \
            f2 = pin[rowm + i]; \
            pin += (NVAL_); \
            f3 = pin[row + ip]; \
            pin += (NVAL_); \
            f4 = pin[rowp + i]; \
            pin += (NVAL_); \
            f5 = pin[rowm + im]; \
            pin += (NVAL_); \
            f6 = pin[rowm + ip]; \
            pin += (NVAL_); \
            f7 = pin[rowp + ip]; \
            pin += (NVAL_); \
            f8 = pin[rowp + im]; \
        } \
        LBM_COLLIDE_STORE_OMEGA(f_out, (NVAL_), idx, omega, f0, f1, f2, f3, f4, f5, f6, f7, f8); \
        return; \
    }

#define LBM_POWER2_YFAST_CASE(SZ_, SHIFT_, NVAL_) \
    if (NX == (SZ_) && NY == (SZ_)) { \
        const uint row = j << (SHIFT_); \
        const uint idx = row + i; \
        const uint mask = (SZ_) - 1u; \
        const uint im = (i - 1u) & mask; \
        const uint ip = (i + 1u) & mask; \
        uint rowm; \
        uint rowp; \
        if (j - 1u < ((SZ_) - 2u)) { \
            rowm = row - (SZ_); \
            rowp = row + (SZ_); \
        } else { \
            rowm = ((j - 1u) & mask) << (SHIFT_); \
            rowp = ((j + 1u) & mask) << (SHIFT_); \
        } \
        device const float *pin = f_in; \
        const float f0 = pin[idx]; \
        pin += (NVAL_); \
        const float f1 = pin[row + im]; \
        pin += (NVAL_); \
        const float f2 = pin[rowm + i]; \
        pin += (NVAL_); \
        const float f3 = pin[row + ip]; \
        pin += (NVAL_); \
        const float f4 = pin[rowp + i]; \
        pin += (NVAL_); \
        const float f5 = pin[rowm + im]; \
        pin += (NVAL_); \
        const float f6 = pin[rowm + ip]; \
        pin += (NVAL_); \
        const float f7 = pin[rowp + ip]; \
        pin += (NVAL_); \
        const float f8 = pin[rowp + im]; \
        LBM_COLLIDE_STORE_OMEGA(f_out, (NVAL_), idx, omega, f0, f1, f2, f3, f4, f5, f6, f7, f8); \
        return; \
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

    const float omega = 1.0f / tau;

    LBM_POWER2_FULL_CASE(256u, 8, 65536u)
    LBM_POWER2_YFAST_CASE(128u, 7, 16384u)
    LBM_POWER2_YFAST_CASE(64u, 6, 4096u)

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

    LBM_COLLIDE_STORE_OMEGA(f_out, N, idx, omega, f0, f1, f2, f3, f4, f5, f6, f7, f8);
}
```

Result of previous attempt:
          64x64_50: correct, 0.51 ms, 28.9 GB/s (effective, 72 B/cell) (14.5% of 200 GB/s)
       128x128_100: correct, 2.04 ms, 57.9 GB/s (effective, 72 B/cell) (28.9% of 200 GB/s)
       256x256_100: correct, 2.41 ms, 196.1 GB/s (effective, 72 B/cell) (98.1% of 200 GB/s)
  score (gmean of fraction): 0.3450

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

#define LBM_COLLIDE_STORE(FOUT_, N_, IDX_, TAU_, F0_, F1_, F2_, F3_, F4_, F5_, F6_, F7_, F8_) do { \
    const float rho = ((((F0_) + (F1_)) + ((F2_) + (F3_))) + (((F4_) + (F5_)) + ((F6_) + (F7_)))) + (F8_); \
    const float mx = (((((F1_) - (F3_)) + (F5_)) - (F6_)) - (F7_)) + (F8_); \
    const float my = (((((F2_) - (F4_)) + (F5_)) + (F6_)) - (F7_)) - (F8_); \
    const float inv_rho = 1.0f / rho; \
    const float mx2 = mx * mx; \
    const float my2 = my * my; \
    const float base = fma(-1.5f, (mx2 + my2) * inv_rho, rho); \
    const float q45 = 4.5f * inv_rho; \
    const float omega = 1.0f / (TAU_); \
    device float *pout = (FOUT_); \
    pout[(IDX_)] = fma(omega, (4.0f / 9.0f) * base - (F0_), (F0_)); \
    const float tx = 3.0f * mx; \
    const float ty = 3.0f * my; \
    const float ex = fma(q45, mx2, base); \
    const float ey = fma(q45, my2, base); \
    pout += (N_); \
    pout[(IDX_)] = fma(omega, (1.0f / 9.0f) * (ex + tx) - (F1_), (F1_)); \
    pout += (N_); \
    pout[(IDX_)] = fma(omega, (1.0f / 9.0f) * (ey + ty) - (F2_), (F2_)); \
    pout += (N_); \
    pout[(IDX_)] = fma(omega, (1.0f / 9.0f) * (ex - tx) - (F3_), (F3_)); \
    pout += (N_); \
    pout[(IDX_)] = fma(omega, (1.0f / 9.0f) * (ey - ty) - (F4_), (F4_)); \
    const float mp = mx + my; \
    const float mm = mx - my; \
    const float mp2 = mp * mp; \
    const float mm2 = mm * mm; \
    const float ep = fma(q45, mp2, base); \
    const float em = fma(q45, mm2, base); \
    const float tp = 3.0f * mp; \
    const float tm = 3.0f * mm; \
    pout += (N_); \
    pout[(IDX_)] = fma(omega, (1.0f / 36.0f) * (ep + tp) - (F5_), (F5_)); \
    pout += (N_); \
    pout[(IDX_)] = fma(omega, (1.0f / 36.0f) * (em - tm) - (F6_), (F6_)); \
    pout += (N_); \
    pout[(IDX_)] = fma(omega, (1.0f / 36.0f) * (ep - tp) - (F7_), (F7_)); \
    pout += (N_); \
    pout[(IDX_)] = fma(omega, (1.0f / 36.0f) * (em + tm) - (F8_), (F8_)); \
} while (0)

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
        const uint row = j << 8;
        const uint idx = row + i;

        float f0, f1, f2, f3, f4, f5, f6, f7, f8;

        if ((i - 1u < 254u) && (j - 1u < 254u)) {
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

        LBM_COLLIDE_STORE(f_out, 65536u, idx, tau, f0, f1, f2, f3, f4, f5, f6, f7, f8);
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

Incumbent result:
          64x64_50: correct, 0.26 ms, 57.0 GB/s (effective, 72 B/cell) (28.5% of 200 GB/s)
       128x128_100: correct, 1.20 ms, 98.4 GB/s (effective, 72 B/cell) (49.2% of 200 GB/s)
       256x256_100: correct, 2.32 ms, 203.1 GB/s (effective, 72 B/cell) (101.5% of 200 GB/s)
  score (gmean of fraction): 0.5221

## History

- iter  7: compile=OK | correct=True | score=0.3515718922758411
- iter  8: compile=OK | correct=True | score=0.3518085650323089
- iter  9: compile=OK | correct=True | score=0.30942080087644286
- iter 10: compile=OK | correct=True | score=0.37110178168958935
- iter 11: compile=OK | correct=True | score=0.4178818116586849
- iter 12: compile=OK | correct=True | score=0.3432721084328763
- iter 13: compile=OK | correct=True | score=0.5220602073808693
- iter 14: compile=OK | correct=True | score=0.34499418179755864

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
