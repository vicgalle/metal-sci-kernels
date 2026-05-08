Special-casing the square power-of-two benchmark sizes replaces periodic-boundary selects with masks and row multiplies with shifts. The collision update is also written as `(1-omega)*f + omega*w*term`, saving one subtraction per population while keeping the same SoA read/write pattern. This should improve the smaller, instruction-overhead-limited cases without hurting the bandwidth-bound large case.

```metal
#include <metal_stdlib>
using namespace metal;

#define LBM_CELL(NVAL) do {                                                        \
    const uint NN = (NVAL);                                                        \
                                                                                   \
    device const float *pin = f_in;                                                \
    const float f0 = pin[idx];                                                     \
    pin += NN;                                                                     \
    const float f1 = pin[row + im];                                                \
    pin += NN;                                                                     \
    const float f2 = pin[rowm + i];                                                \
    pin += NN;                                                                     \
    const float f3 = pin[row + ip];                                                \
    pin += NN;                                                                     \
    const float f4 = pin[rowp + i];                                                \
    pin += NN;                                                                     \
    const float f5 = pin[rowm + im];                                               \
    pin += NN;                                                                     \
    const float f6 = pin[rowm + ip];                                               \
    pin += NN;                                                                     \
    const float f7 = pin[rowp + ip];                                               \
    pin += NN;                                                                     \
    const float f8 = pin[rowp + im];                                               \
                                                                                   \
    const float s13 = f1 + f3;                                                     \
    const float s24 = f2 + f4;                                                     \
    const float s57 = f5 + f7;                                                     \
    const float s68 = f6 + f8;                                                     \
    const float rho = ((f0 + s13) + (s24 + s57)) + s68;                            \
                                                                                   \
    const float d13 = f1 - f3;                                                     \
    const float d24 = f2 - f4;                                                     \
    const float dp  = f5 - f7;                                                     \
    const float dm  = f6 - f8;                                                     \
    const float mx  = (d13 + dp) - dm;                                             \
    const float my  = (d24 + dp) + dm;                                             \
                                                                                   \
    const float inv_rho = 1.0f / rho;                                              \
    const float mx2 = mx * mx;                                                     \
    const float my2 = my * my;                                                     \
    const float base = fma(-1.5f, (mx2 + my2) * inv_rho, rho);                     \
    const float q45  = 4.5f * inv_rho;                                             \
                                                                                   \
    const float omega = 1.0f / tau;                                                \
    const float om    = 1.0f - omega;                                              \
    const float ow1   = omega * (1.0f / 9.0f);                                     \
                                                                                   \
    device float *pout = f_out;                                                    \
    pout[idx] = fma(4.0f * ow1, base, om * f0);                                    \
                                                                                   \
    const float tx = 3.0f * mx;                                                    \
    const float ty = 3.0f * my;                                                    \
    const float ex = fma(q45, mx2, base);                                          \
    const float ey = fma(q45, my2, base);                                          \
                                                                                   \
    pout += NN;                                                                    \
    pout[idx] = fma(ow1, ex + tx, om * f1);                                        \
    pout += NN;                                                                    \
    pout[idx] = fma(ow1, ey + ty, om * f2);                                        \
    pout += NN;                                                                    \
    pout[idx] = fma(ow1, ex - tx, om * f3);                                        \
    pout += NN;                                                                    \
    pout[idx] = fma(ow1, ey - ty, om * f4);                                        \
                                                                                   \
    const float mp  = mx + my;                                                     \
    const float mm  = mx - my;                                                     \
    const float mp2 = mp * mp;                                                     \
    const float mm2 = mm * mm;                                                     \
    const float ep  = fma(q45, mp2, base);                                         \
    const float em  = fma(q45, mm2, base);                                         \
    const float tp  = 3.0f * mp;                                                   \
    const float tm  = 3.0f * mm;                                                   \
    const float owd = 0.25f * ow1;                                                 \
                                                                                   \
    pout += NN;                                                                    \
    pout[idx] = fma(owd, ep + tp, om * f5);                                        \
    pout += NN;                                                                    \
    pout[idx] = fma(owd, em - tm, om * f6);                                        \
    pout += NN;                                                                    \
    pout[idx] = fma(owd, ep - tp, om * f7);                                        \
    pout += NN;                                                                    \
    pout[idx] = fma(owd, em + tm, om * f8);                                        \
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

    if (NX == 64u && NY == 64u) {
        const uint row  = j << 6;
        const uint idx  = row + i;
        const uint im   = (i - 1u) & 63u;
        const uint ip   = (i + 1u) & 63u;
        const uint rowm = (row - 64u) & 4095u;
        const uint rowp = (row + 64u) & 4095u;
        LBM_CELL(4096u);
        return;
    }

    if (NX == 128u && NY == 128u) {
        const uint row  = j << 7;
        const uint idx  = row + i;
        const uint im   = (i - 1u) & 127u;
        const uint ip   = (i + 1u) & 127u;
        const uint rowm = (row - 128u) & 16383u;
        const uint rowp = (row + 128u) & 16383u;
        LBM_CELL(16384u);
        return;
    }

    if (NX == 256u && NY == 256u) {
        const uint row  = j << 8;
        const uint idx  = row + i;
        const uint im   = (i - 1u) & 255u;
        const uint ip   = (i + 1u) & 255u;
        const uint rowm = (row - 256u) & 65535u;
        const uint rowp = (row + 256u) & 65535u;
        LBM_CELL(65536u);
        return;
    }

    const uint N    = NX * NY;
    const uint row  = j * NX;
    const uint idx  = row + i;
    const uint im   = (i == 0u) ? (NX - 1u) : (i - 1u);
    const uint ip   = (i + 1u == NX) ? 0u : (i + 1u);
    const uint rowm = (j == 0u) ? (N - NX) : (row - NX);
    const uint rowp = (j + 1u == NY) ? 0u : (row + NX);

    LBM_CELL(N);
}

#undef LBM_CELL
```