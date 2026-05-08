I’m adding exact 64×64 and 128×128 power-of-two fast paths using constant shifts/masks for periodic addressing, avoiding per-thread boundary selects and runtime multiplies on the smaller benchmarks. The 256×256 path is kept in the incumbent form to preserve its near-roofline behavior. This should mainly improve the launch/compute-sensitive small cases while minimizing risk to the already excellent large case.

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

#define LBM_POWER2_PATH(SHIFT_, MASK_, NCONST_) do { \
    const uint row = j << (SHIFT_); \
    const uint idx = row + i; \
    const uint im = (i - 1u) & (MASK_); \
    const uint ip = (i + 1u) & (MASK_); \
    const uint rowm = ((j - 1u) & (MASK_)) << (SHIFT_); \
    const uint rowp = ((j + 1u) & (MASK_)) << (SHIFT_); \
    device const float *pin = f_in; \
    const float f0 = pin[idx]; \
    pin += (NCONST_); \
    const float f1 = pin[row + im]; \
    pin += (NCONST_); \
    const float f2 = pin[rowm + i]; \
    pin += (NCONST_); \
    const float f3 = pin[row + ip]; \
    pin += (NCONST_); \
    const float f4 = pin[rowp + i]; \
    pin += (NCONST_); \
    const float f5 = pin[rowm + im]; \
    pin += (NCONST_); \
    const float f6 = pin[rowm + ip]; \
    pin += (NCONST_); \
    const float f7 = pin[rowp + ip]; \
    pin += (NCONST_); \
    const float f8 = pin[rowp + im]; \
    LBM_COLLIDE_STORE(f_out, (NCONST_), idx, tau, f0, f1, f2, f3, f4, f5, f6, f7, f8); \
    return; \
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

    if (NX == 64u && NY == 64u) {
        LBM_POWER2_PATH(6u, 63u, 4096u);
    }

    if (NX == 128u && NY == 128u) {
        LBM_POWER2_PATH(7u, 127u, 16384u);
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