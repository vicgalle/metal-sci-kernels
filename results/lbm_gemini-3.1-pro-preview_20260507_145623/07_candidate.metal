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
    
    // Fast periodic boundary wrapping without modulo
    uint x_m1 = (i == 0) ? NX - 1 : i - 1;
    uint x_p1 = (i + 1 == NX) ? 0 : i + 1;
    uint y_m1 = (j == 0) ? NY - 1 : j - 1;
    uint y_p1 = (j + 1 == NY) ? 0 : j + 1;

    uint base    = j * NX;
    uint base_m1 = y_m1 * NX;
    uint base_p1 = y_p1 * NX;

    // Fully unrolled pull streaming loads
    float f0 = f_in[base + i];
    float f1 = f_in[base + x_m1 + N];
    float f2 = f_in[base_m1 + i + 2 * N];
    float f3 = f_in[base + x_p1 + 3 * N];
    float f4 = f_in[base_p1 + i + 4 * N];
    float f5 = f_in[base_m1 + x_m1 + 5 * N];
    float f6 = f_in[base_m1 + x_p1 + 6 * N];
    float f7 = f_in[base_p1 + x_p1 + 7 * N];
    float f8 = f_in[base_p1 + x_m1 + 8 * N];

    // Macroscopic moments
    float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    float ux = f1 - f3 + f5 - f6 - f7 + f8;
    float uy = f2 - f4 + f5 + f6 - f7 - f8;

    float inv_rho = 1.0f / rho;
    ux *= inv_rho;
    uy *= inv_rho;

    // BGK Collision
    float usq = ux * ux + uy * uy;
    float usq_term = 1.0f - 1.5f * usq;
    float inv_tau = 1.0f / tau;
    
    float w_rho_0 = (4.0f / 9.0f) * rho;
    float w_rho_1 = (1.0f / 9.0f) * rho;
    float w_rho_5 = (1.0f / 36.0f) * rho;

    float feq0 = w_rho_0 * usq_term;
    float out0 = f0 - inv_tau * (f0 - feq0);

    float cu1 = ux;
    float feq1 = w_rho_1 * (usq_term + cu1 * (3.0f + 4.5f * cu1));
    float out1 = f1 - inv_tau * (f1 - feq1);

    float cu2 = uy;
    float feq2 = w_rho_1 * (usq_term + cu2 * (3.0f + 4.5f * cu2));
    float out2 = f2 - inv_tau * (f2 - feq2);

    float cu3 = -ux;
    float feq3 = w_rho_1 * (usq_term + cu3 * (3.0f + 4.5f * cu3));
    float out3 = f3 - inv_tau * (f3 - feq3);

    float cu4 = -uy;
    float feq4 = w_rho_1 * (usq_term + cu4 * (3.0f + 4.5f * cu4));
    float out4 = f4 - inv_tau * (f4 - feq4);

    float cu5 = ux + uy;
    float feq5 = w_rho_5 * (usq_term + cu5 * (3.0f + 4.5f * cu5));
    float out5 = f5 - inv_tau * (f5 - feq5);

    float cu6 = -ux + uy;
    float feq6 = w_rho_5 * (usq_term + cu6 * (3.0f + 4.5f * cu6));
    float out6 = f6 - inv_tau * (f6 - feq6);

    float cu7 = -cu5;
    float feq7 = w_rho_5 * (usq_term + cu7 * (3.0f + 4.5f * cu7));
    float out7 = f7 - inv_tau * (f7 - feq7);

    float cu8 = -cu6;
    float feq8 = w_rho_5 * (usq_term + cu8 * (3.0f + 4.5f * cu8));
    float out8 = f8 - inv_tau * (f8 - feq8);

    // Fully unrolled stores
    uint out_idx = base + i;
    f_out[out_idx]         = out0;
    f_out[out_idx + N]     = out1;
    f_out[out_idx + 2 * N] = out2;
    f_out[out_idx + 3 * N] = out3;
    f_out[out_idx + 4 * N] = out4;
    f_out[out_idx + 5 * N] = out5;
    f_out[out_idx + 6 * N] = out6;
    f_out[out_idx + 7 * N] = out7;
    f_out[out_idx + 8 * N] = out8;
}