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