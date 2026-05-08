#include <metal_stdlib>
using namespace metal;

#define LBM_CONST_BODY(SH, MASK, NVAL)                                      \
do {                                                                         \
    const uint row  = j << (SH);                                             \
    const uint idx  = row | i;                                               \
    const uint im   = (i - 1u) & (MASK);                                     \
    const uint ip   = (i + 1u) & (MASK);                                     \
    const uint rowm = ((j - 1u) & (MASK)) << (SH);                           \
    const uint rowp = ((j + 1u) & (MASK)) << (SH);                           \
                                                                             \
    const float f0 = f_in[idx];                                              \
    const float f1 = f_in[(NVAL)       + (row  | im)];                       \
    const float f2 = f_in[(NVAL) * 2u  + (rowm | i )];                       \
    const float f3 = f_in[(NVAL) * 3u  + (row  | ip)];                       \
    const float f4 = f_in[(NVAL) * 4u  + (rowp | i )];                       \
    const float f5 = f_in[(NVAL) * 5u  + (rowm | im)];                       \
    const float f6 = f_in[(NVAL) * 6u  + (rowm | ip)];                       \
    const float f7 = f_in[(NVAL) * 7u  + (rowp | ip)];                       \
    const float f8 = f_in[(NVAL) * 8u  + (rowp | im)];                       \
                                                                             \
    const float s13 = f1 + f3;                                               \
    const float s24 = f2 + f4;                                               \
    const float s57 = f5 + f7;                                               \
    const float s68 = f6 + f8;                                               \
    const float rho = ((f0 + s13) + (s24 + s57)) + s68;                      \
                                                                             \
    const float d13 = f1 - f3;                                               \
    const float d24 = f2 - f4;                                               \
    const float d57 = f5 - f7;                                               \
    const float d68 = f6 - f8;                                               \
    const float mx  = (d13 + d57) - d68;                                     \
    const float my  = (d24 + d57) + d68;                                     \
                                                                             \
    const float inv_rho = 1.0f / rho;                                        \
    const float mx2 = mx * mx;                                               \
    const float my2 = my * my;                                               \
    const float m2r = (mx2 + my2) * inv_rho;                                 \
    const float base = fma(-1.5f, m2r, rho);                                 \
    const float q45  = 4.5f * inv_rho;                                       \
                                                                             \
    const float omega = 1.0f / tau;                                          \
    const float om    = 1.0f - omega;                                        \
    const float ow1   = omega * (1.0f / 9.0f);                               \
                                                                             \
    f_out[idx] = fma(4.0f * ow1, base, om * f0);                             \
                                                                             \
    const float tx = 3.0f * mx;                                              \
    const float ty = 3.0f * my;                                              \
    const float ex = fma(q45, mx2, base);                                    \
    const float ey = fma(q45, my2, base);                                    \
                                                                             \
    f_out[(NVAL)       + idx] = fma(ow1, ex + tx, om * f1);                  \
    f_out[(NVAL) * 2u  + idx] = fma(ow1, ey + ty, om * f2);                  \
    f_out[(NVAL) * 3u  + idx] = fma(ow1, ex - tx, om * f3);                  \
    f_out[(NVAL) * 4u  + idx] = fma(ow1, ey - ty, om * f4);                  \
                                                                             \
    const float diag0 = fma(3.0f, m2r, rho);                                 \
    const float cross = (9.0f * inv_rho) * (mx * my);                        \
    const float ep = diag0 + cross;                                          \
    const float em = diag0 - cross;                                          \
    const float tp = tx + ty;                                                \
    const float tm = tx - ty;                                                \
    const float owd = 0.25f * ow1;                                           \
                                                                             \
    f_out[(NVAL) * 5u  + idx] = fma(owd, ep + tp, om * f5);                  \
    f_out[(NVAL) * 6u  + idx] = fma(owd, em - tm, om * f6);                  \
    f_out[(NVAL) * 7u  + idx] = fma(owd, ep - tp, om * f7);                  \
    f_out[(NVAL) * 8u  + idx] = fma(owd, em + tm, om * f8);                  \
    return;                                                                  \
} while (false)

kernel void lbm_step(device const float *f_in   [[buffer(0)]],
                     device       float *f_out  [[buffer(1)]],
                     constant uint        &NX   [[buffer(2)]],
                     constant uint        &NY   [[buffer(3)]],
                     constant float       &tau  [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x;
    const uint j = gid.y;
    if (i >= NX || j >= NY) return;

    if (NX == NY) {
        switch (NX) {
            case 64u:
                LBM_CONST_BODY(6u, 63u, 4096u);
            case 128u:
                LBM_CONST_BODY(7u, 127u, 16384u);
            case 256u:
                LBM_CONST_BODY(8u, 255u, 65536u);
            default:
                break;
        }
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

#undef LBM_CONST_BODY