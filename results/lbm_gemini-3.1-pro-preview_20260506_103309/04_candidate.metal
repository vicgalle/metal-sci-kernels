#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(256)]]
kernel void lbm_step(device const float *f_in   [[buffer(0)]],
                     device       float *f_out  [[buffer(1)]],
                     constant uint        &NX   [[buffer(2)]],
                     constant uint        &NY   [[buffer(3)]],
                     constant float       &tau  [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint j = gid.y;
    if (i >= NX || j >= NY) return;

    uint N = NX * NY;
    uint j_NX = j * NX;

    // Fast periodic boundary wrapping
    uint ip1 = (i + 1 == NX) ? 0 : i + 1;
    uint im1 = (i == 0) ? NX - 1 : i - 1;
    uint jp1_NX = (j + 1 == NY) ? 0 : j_NX + NX;
    uint jm1_NX = (j == 0) ? N - NX : j_NX - NX;

    // 1. Pull streaming
    float f0 = f_in[j_NX + i];
    float f1 = f_in[N + j_NX + im1];
    float f2 = f_in[2 * N + jm1_NX + i];
    float f3 = f_in[3 * N + j_NX + ip1];
    float f4 = f_in[4 * N + jp1_NX + i];
    float f5 = f_in[5 * N + jm1_NX + im1];
    float f6 = f_in[6 * N + jm1_NX + ip1];
    float f7 = f_in[7 * N + jp1_NX + ip1];
    float f8 = f_in[8 * N + jp1_NX + im1];

    // 2. Moments computation (symmetric reduction)
    float d13 = f1 - f3;
    float d57 = f5 - f7;
    float d68 = f6 - f8;
    float d24 = f2 - f4;

    float s13 = f1 + f3;
    float s57 = f5 + f7;
    float s68 = f6 + f8;
    float s24 = f2 + f4;

    float rho = f0 + s13 + s24 + s57 + s68;
    float inv_rho = 1.0f / rho;

    float ux = (d13 + d57 - d68) * inv_rho;
    float uy = (d24 + d57 + d68) * inv_rho;

    // 3. BGK collision preparation
    float sq_x = ux * ux;
    float sq_y = uy * uy;
    float om_usq15 = 1.0f - 1.5f * (sq_x + sq_y);

    float inv_tau_val = 1.0f / tau;
    float om_inv_tau = 1.0f - inv_tau_val;

    float T = inv_tau_val * rho;
    float w0_T = T * (4.0f / 9.0f);
    float w1_T = T * (1.0f / 9.0f);
    float w5_T = T * (1.0f / 36.0f);

    uint idx = j_NX + i;

    // Direction 0
    f_out[idx] = fma(f0, om_inv_tau, w0_T * om_usq15);

    // Directions 1 and 3 (ux and -ux)
    float term_x = fma(sq_x, 4.5f, om_usq15);
    float lin_x  = ux * 3.0f;
    f_out[N + idx]     = fma(f1, om_inv_tau, w1_T * (term_x + lin_x));
    f_out[3 * N + idx] = fma(f3, om_inv_tau, w1_T * (term_x - lin_x));

    // Directions 2 and 4 (uy and -uy)
    float term_y = fma(sq_y, 4.5f, om_usq15);
    float lin_y  = uy * 3.0f;
    f_out[2 * N + idx] = fma(f2, om_inv_tau, w1_T * (term_y + lin_y));
    f_out[4 * N + idx] = fma(f4, om_inv_tau, w1_T * (term_y - lin_y));

    // Directions 5 and 7 (cu5 and -cu5)
    float cu5 = ux + uy;
    float sq_5 = cu5 * cu5;
    float term_5 = fma(sq_5, 4.5f, om_usq15);
    float lin_5  = cu5 * 3.0f;
    f_out[5 * N + idx] = fma(f5, om_inv_tau, w5_T * (term_5 + lin_5));
    f_out[7 * N + idx] = fma(f7, om_inv_tau, w5_T * (term_5 - lin_5));

    // Directions 6 and 8 (cu6 and -cu6)
    float cu6 = -ux + uy;
    float sq_6 = cu6 * cu6;
    float term_6 = fma(sq_6, 4.5f, om_usq15);
    float lin_6  = cu6 * 3.0f;
    f_out[6 * N + idx] = fma(f6, om_inv_tau, w5_T * (term_6 + lin_6));
    f_out[8 * N + idx] = fma(f8, om_inv_tau, w5_T * (term_6 - lin_6));
}