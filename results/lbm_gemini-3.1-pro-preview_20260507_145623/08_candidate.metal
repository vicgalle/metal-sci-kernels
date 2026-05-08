#include <metal_stdlib>
using namespace metal;

kernel void lbm_step(device const float *f_in   [[buffer(0)]],
                     device       float *f_out  [[buffer(1)]],
                     constant uint        &NX   [[buffer(2)]],
                     constant uint        &NY   [[buffer(3)]],
                     constant float       &tau  [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]]) 
{
    uint i = gid.x;
    uint j = gid.y;
    if (i >= NX || j >= NY) return;

    uint N = NX * NY;

    // Fast periodic boundary wrapping using select instead of modulo
    uint x_m1 = (i == 0) ? NX - 1 : i - 1;
    uint x_p1 = (i + 1 == NX) ? 0 : i + 1;
    uint y_m1 = (j == 0) ? NY - 1 : j - 1;
    uint y_p1 = (j + 1 == NY) ? 0 : j + 1;

    // Precompute spatial base offsets
    uint base_0  = j * NX;
    uint base_m1 = y_m1 * NX;
    uint base_p1 = y_p1 * NX;

    // Precompute channel offsets
    uint ch1 = N;
    uint ch2 = 2 * N;
    uint ch3 = 3 * N;
    uint ch4 = 4 * N;
    uint ch5 = 5 * N;
    uint ch6 = 6 * N;
    uint ch7 = 7 * N;
    uint ch8 = 8 * N;

    // Fully unrolled pull streaming loads
    float f0 = f_in[base_0  + i];
    float f1 = f_in[ch1 + base_0  + x_m1];
    float f2 = f_in[ch2 + base_m1 + i];
    float f3 = f_in[ch3 + base_0  + x_p1];
    float f4 = f_in[ch4 + base_p1 + i];
    float f5 = f_in[ch5 + base_m1 + x_m1];
    float f6 = f_in[ch6 + base_m1 + x_p1];
    float f7 = f_in[ch7 + base_p1 + x_p1];
    float f8 = f_in[ch8 + base_p1 + x_m1];

    // Macroscopic moments
    float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    float ux  = f1 - f3 + f5 - f6 - f7 + f8;
    float uy  = f2 - f4 + f5 + f6 - f7 - f8;

    float inv_rho = 1.0f / rho;
    ux *= inv_rho;
    uy *= inv_rho;

    // Shared collision terms
    float usq = ux * ux + uy * uy;
    float usq_term = 1.0f - 1.5f * usq;
    float inv_tau = 1.0f / tau;
    uint idx = base_0 + i;

    float w_rho_0 = (4.0f / 9.0f) * rho;
    float w_rho_1 = (1.0f / 9.0f) * rho;
    float w_rho_5 = (1.0f / 36.0f) * rho;

    // Center direction
    f_out[idx] = f0 - inv_tau * (f0 - w_rho_0 * usq_term);

    // Pair 1 & 3 (Horizontal)
    float cu1 = ux;
    float usq_plus_sq1 = usq_term + 4.5f * (cu1 * cu1);
    f_out[ch1 + idx] = f1 - inv_tau * (f1 - w_rho_1 * (usq_plus_sq1 + 3.0f * cu1));
    f_out[ch3 + idx] = f3 - inv_tau * (f3 - w_rho_1 * (usq_plus_sq1 - 3.0f * cu1));

    // Pair 2 & 4 (Vertical)
    float cu2 = uy;
    float usq_plus_sq2 = usq_term + 4.5f * (cu2 * cu2);
    f_out[ch2 + idx] = f2 - inv_tau * (f2 - w_rho_1 * (usq_plus_sq2 + 3.0f * cu2));
    f_out[ch4 + idx] = f4 - inv_tau * (f4 - w_rho_1 * (usq_plus_sq2 - 3.0f * cu2));

    // Pair 5 & 7 (Diagonal top-right to bottom-left)
    float cu5 = ux + uy;
    float usq_plus_sq5 = usq_term + 4.5f * (cu5 * cu5);
    f_out[ch5 + idx] = f5 - inv_tau * (f5 - w_rho_5 * (usq_plus_sq5 + 3.0f * cu5));
    f_out[ch7 + idx] = f7 - inv_tau * (f7 - w_rho_5 * (usq_plus_sq5 - 3.0f * cu5));

    // Pair 6 & 8 (Diagonal top-left to bottom-right)
    float cu6 = -ux + uy;
    float usq_plus_sq6 = usq_term + 4.5f * (cu6 * cu6);
    f_out[ch6 + idx] = f6 - inv_tau * (f6 - w_rho_5 * (usq_plus_sq6 + 3.0f * cu6));
    f_out[ch8 + idx] = f8 - inv_tau * (f8 - w_rho_5 * (usq_plus_sq6 - 3.0f * cu6));
}