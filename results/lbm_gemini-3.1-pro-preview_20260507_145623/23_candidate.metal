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

    int inx = int(nx);
    int iny = int(ny);
    uint N = nx * ny;
    uint idx = j * nx + i;

    // Fast 1D periodic boundary offset calculation
    int im1 = (i == 0) ? (inx - 1) : -1;
    int ip1 = (i == nx - 1) ? (1 - inx) : 1;
    int jm1 = (j == 0) ? (inx * (iny - 1)) : -inx;
    int jp1 = (j == ny - 1) ? -(inx * (iny - 1)) : inx;

    // PULL streaming reads
    float f[9];
    f[0] = f_in[idx];
    f[1] = f_in[N + idx + uint(im1)];
    f[2] = f_in[2 * N + idx + uint(jm1)];
    f[3] = f_in[3 * N + idx + uint(ip1)];
    f[4] = f_in[4 * N + idx + uint(jp1)];
    f[5] = f_in[5 * N + idx + uint(im1 + jm1)];
    f[6] = f_in[6 * N + idx + uint(ip1 + jm1)];
    f[7] = f_in[7 * N + idx + uint(ip1 + jp1)];
    f[8] = f_in[8 * N + idx + uint(im1 + jp1)];

    // Tree-reduction for moments
    float rho = (f[0] + f[1]) + (f[2] + f[3]) + (f[4] + f[5]) + (f[6] + f[7]) + f[8];
    float ux = (f[1] + f[5] + f[8]) - (f[3] + f[6] + f[7]);
    float uy = (f[2] + f[5] + f[6]) - (f[4] + f[7] + f[8]);

    float inv_rho = 1.0f / rho;
    ux *= inv_rho;
    uy *= inv_rho;

    // BGK collision setup
    float usq = ux * ux + uy * uy;
    float inv_tau_val = 1.0f / tau;
    float oma = 1.0f - inv_tau_val;

    float rt = rho * inv_tau_val;
    float term0 = 1.0f - 1.5f * usq;

    float w0_rt = (4.0f / 9.0f) * rt;
    float w1_rt = (1.0f / 9.0f) * rt;
    float w5_rt = (1.0f / 36.0f) * rt;

    // Symmetric velocity combinations
    float ux_sq = ux * ux;
    float uy_sq = uy * uy;
    float cu5 = ux + uy;
    float cu6 = -ux + uy;
    float cu5_sq = cu5 * cu5;
    float cu6_sq = cu6 * cu6;

    // Factored equilibrium terms
    float term_x = w1_rt * (term0 + 4.5f * ux_sq);
    float term_y = w1_rt * (term0 + 4.5f * uy_sq);
    float term_xy = w5_rt * (term0 + 4.5f * cu5_sq);
    float term_xmy = w5_rt * (term0 + 4.5f * cu6_sq);

    float ux_w1 = w1_rt * 3.0f * ux;
    float uy_w1 = w1_rt * 3.0f * uy;
    float cu5_w5 = w5_rt * 3.0f * cu5;
    float cu6_w5 = w5_rt * 3.0f * cu6;

    // Ordered writes to memory
    f_out[idx]         = f[0] * oma + w0_rt * term0;
    f_out[N + idx]     = f[1] * oma + term_x + ux_w1;
    f_out[2 * N + idx] = f[2] * oma + term_y + uy_w1;
    f_out[3 * N + idx] = f[3] * oma + term_x - ux_w1;
    f_out[4 * N + idx] = f[4] * oma + term_y - uy_w1;
    f_out[5 * N + idx] = f[5] * oma + term_xy + cu5_w5;
    f_out[6 * N + idx] = f[6] * oma + term_xmy + cu6_w5;
    f_out[7 * N + idx] = f[7] * oma + term_xy - cu5_w5;
    f_out[8 * N + idx] = f[8] * oma + term_xmy - cu6_w5;
}