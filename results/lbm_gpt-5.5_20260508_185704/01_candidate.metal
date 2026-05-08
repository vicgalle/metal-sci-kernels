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

    const uint N  = NX * NY;
    const uint n1 = N;
    const uint n2 = n1 + N;
    const uint n3 = n2 + N;
    const uint n4 = n3 + N;
    const uint n5 = n4 + N;
    const uint n6 = n5 + N;
    const uint n7 = n6 + N;
    const uint n8 = n7 + N;

    const uint idx = j * NX + i;

    float f0, f1, f2, f3, f4, f5, f6, f7, f8;

    if (i > 0u && i + 1u < NX && j > 0u && j + 1u < NY) {
        f0 = f_in[idx];
        f1 = f_in[n1 + idx - 1u];
        f2 = f_in[n2 + idx - NX];
        f3 = f_in[n3 + idx + 1u];
        f4 = f_in[n4 + idx + NX];
        f5 = f_in[n5 + idx - NX - 1u];
        f6 = f_in[n6 + idx - NX + 1u];
        f7 = f_in[n7 + idx + NX + 1u];
        f8 = f_in[n8 + idx + NX - 1u];
    } else {
        const uint im = (i == 0u) ? (NX - 1u) : (i - 1u);
        const uint ip = (i + 1u == NX) ? 0u : (i + 1u);
        const uint jm = (j == 0u) ? (NY - 1u) : (j - 1u);
        const uint jp = (j + 1u == NY) ? 0u : (j + 1u);

        const uint row  = j  * NX;
        const uint rowm = jm * NX;
        const uint rowp = jp * NX;

        f0 = f_in[row + i];
        f1 = f_in[n1 + row  + im];
        f2 = f_in[n2 + rowm + i ];
        f3 = f_in[n3 + row  + ip];
        f4 = f_in[n4 + rowp + i ];
        f5 = f_in[n5 + rowm + im];
        f6 = f_in[n6 + rowm + ip];
        f7 = f_in[n7 + rowp + ip];
        f8 = f_in[n8 + rowp + im];
    }

    const float rho = ((((((((f0 + f1) + f2) + f3) + f4) + f5) + f6) + f7) + f8);
    const float mx  = (((((f1 - f3) + f5) - f6) - f7) + f8);
    const float my  = (((((f2 - f4) + f5) + f6) - f7) - f8);

    const float inv_rho = 1.0f / rho;
    const float mx2 = mx * mx;
    const float my2 = my * my;
    const float m2  = mx2 + my2;

    const float base_eq = rho - 1.5f * m2 * inv_rho;

    const float eq_x = base_eq + 4.5f * mx2 * inv_rho;
    const float eq_y = base_eq + 4.5f * my2 * inv_rho;

    const float mp = mx + my;
    const float mm = mx - my;
    const float eq_p = base_eq + 4.5f * (mp * mp) * inv_rho;
    const float eq_m = base_eq + 4.5f * (mm * mm) * inv_rho;

    const float omega = 1.0f / tau;
    const float keep  = 1.0f - omega;

    const float ow0 = omega * (4.0f / 9.0f);
    const float ow1 = omega * (1.0f / 9.0f);
    const float ow5 = omega * (1.0f / 36.0f);

    const float tx = 3.0f * mx;
    const float ty = 3.0f * my;
    const float tp = 3.0f * mp;
    const float tm = 3.0f * mm;

    device float *out = f_out + idx;

    out[0 ] = fma(ow0, base_eq,      keep * f0);
    out[n1] = fma(ow1, eq_x + tx,    keep * f1);
    out[n2] = fma(ow1, eq_y + ty,    keep * f2);
    out[n3] = fma(ow1, eq_x - tx,    keep * f3);
    out[n4] = fma(ow1, eq_y - ty,    keep * f4);
    out[n5] = fma(ow5, eq_p + tp,    keep * f5);
    out[n6] = fma(ow5, eq_m - tm,    keep * f6);
    out[n7] = fma(ow5, eq_p - tp,    keep * f7);
    out[n8] = fma(ow5, eq_m + tm,    keep * f8);
}