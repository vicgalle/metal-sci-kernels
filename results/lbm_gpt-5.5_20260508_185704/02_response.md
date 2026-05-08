This version keeps the pull-streaming fully unrolled but removes the interior/boundary split from the previous attempt, avoiding divergent large code paths and reducing live integer offsets by walking plane pointers. Collision is emitted sequentially with fewer long-lived temporaries, which should improve occupancy/register pressure while preserving coalesced SoA reads and writes.

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