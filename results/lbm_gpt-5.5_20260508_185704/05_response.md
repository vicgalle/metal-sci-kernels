I avoid the prior size-specialized branch/code bloat and keep a single generic path. The kernel reduces scalar work by reusing pair sums/differences for moments, precomputing collision weights, and forming diagonal equilibria from \(m^2\) and \(m_xm_y\) instead of two extra squared sums, while preserving the same SoA memory traffic.

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

    device float *pout = f_out;

    pout[idx] = fma(ow0, base, om * f0);

    pout += N;
    pout[idx] = fma(ow1, ex + tx, om * f1);

    pout += N;
    pout[idx] = fma(ow1, ey + ty, om * f2);

    pout += N;
    pout[idx] = fma(ow1, ex - tx, om * f3);

    pout += N;
    pout[idx] = fma(ow1, ey - ty, om * f4);

    pout += N;
    pout[idx] = fma(owd, ep + tp, om * f5);

    pout += N;
    pout[idx] = fma(owd, em - tm, om * f6);

    pout += N;
    pout[idx] = fma(owd, ep - tp, om * f7);

    pout += N;
    pout[idx] = fma(owd, em + tm, om * f8);
}
```