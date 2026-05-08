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
    uint nx = NX;
    uint ny = NY;
    
    if (i >= nx || j >= ny) return;

    uint N = nx * ny;
    uint idx = j * nx + i;

    // Fast 1D periodic boundary offset calculation
    uint im1 = (i == 0) ? (nx - 1) : -1u;
    uint ip1 = (i == nx - 1) ? (1u - nx) : 1u;
    uint jm1 = (j == 0) ? (nx * (ny - 1)) : -nx;
    uint jp1 = (j == ny - 1) ? -(nx * (ny - 1)) : nx;

    // Ordered PULL stream reads
    float f0 = f_in[idx];
    float f1 = f_in[N + idx + im1];
    float f2 = f_in[2 * N + idx + jm1];
    float f3 = f_in[3 * N + idx + ip1];
    float f4 = f_in[4 * N + idx + jp1];
    float f5 = f_in[5 * N + idx + im1 + jm1];
    float f6 = f_in[6 * N + idx + ip1 + jm1];
    float f7 = f_in[7 * N + idx + ip1 + jp1];
    float f8 = f_in[8 * N + idx + im1 + jp1];

    // Tree-reduction for moments to maximize ILP
    float rho_01 = f0 + f1;
    float rho_23 = f2 + f3;
    float rho_45 = f4 + f5;
    float rho_67 = f6 + f7;
    float rho = (rho_01 + rho_23) + (rho_45 + rho_67) + f8;

    float ux_p = f1 + f5 + f8;
    float ux_n = f3 + f6 + f7;
    float ux = ux_p - ux_n;

    float uy_p = f2 + f5 + f6;
    float uy_n = f4 + f7 + f8;
    float uy = uy_p - uy_n;

    float inv_rho = 1.0f / rho;
    ux *= inv_rho;
    uy *= inv_rho;

    // BGK Collision setup
    float inv_tau_val = 1.0f / tau;
    float oma = 1.0f - inv_tau_val;

    float usq = ux * ux + uy * uy;
    float term0 = 1.0f - 1.5f * usq;

    // Weights
    float rt = rho * inv_tau_val;
    float w0_rt = (4.0f / 9.0f) * rt;
    float w1_rt = (1.0f / 9.0f) * rt;
    float w5_rt = (1.0f / 36.0f) * rt;

    // Collision macroscopic components
    float ux_sq = ux * ux;
    float uy_sq = uy * uy;
    float ux3 = 3.0f * ux;
    float uy3 = 3.0f * uy;

    float term_x = term0 + 4.5f * ux_sq;
    float term_y = term0 + 4.5f * uy_sq;

    float cu5 = ux + uy;
    float cu6 = -ux + uy;
    
    float term_xy = term0 + 4.5f * (cu5 * cu5);
    float term_xmy = term0 + 4.5f * (cu6 * cu6);
    
    float cu5_3 = 3.0f * cu5;
    float cu6_3 = 3.0f * cu6;

    // Ordered writes to global memory to maximize write-combining efficiency
    f_out[idx]         = f0 * oma + w0_rt * term0;
    f_out[N + idx]     = f1 * oma + w1_rt * (term_x + ux3);
    f_out[2 * N + idx] = f2 * oma + w1_rt * (term_y + uy3);
    f_out[3 * N + idx] = f3 * oma + w1_rt * (term_x - ux3);
    f_out[4 * N + idx] = f4 * oma + w1_rt * (term_y - uy3);
    f_out[5 * N + idx] = f5 * oma + w5_rt * (term_xy + cu5_3);
    f_out[6 * N + idx] = f6 * oma + w5_rt * (term_xmy + cu6_3);
    f_out[7 * N + idx] = f7 * oma + w5_rt * (term_xy - cu5_3);
    f_out[8 * N + idx] = f8 * oma + w5_rt * (term_xmy - cu6_3);
}