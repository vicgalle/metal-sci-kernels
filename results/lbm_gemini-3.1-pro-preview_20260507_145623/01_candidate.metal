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
    
    // Efficient periodic boundary wrap
    uint ip1 = (i == NX - 1) ? 0 : i + 1;
    uint im1 = (i == 0) ? NX - 1 : i - 1;
    uint jp1 = (j == NY - 1) ? 0 : j + 1;
    uint jm1 = (j == 0) ? NY - 1 : j - 1;

    // Precalculate row offsets
    uint jNX = j * NX;
    uint jm1NX = jm1 * NX;
    uint jp1NX = jp1 * NX;

    // Determine fetch indices for each streamed direction
    uint idx0 = jNX + i;
    uint idx1 = jNX + im1;
    uint idx2 = jm1NX + i;
    uint idx3 = jNX + ip1;
    uint idx4 = jp1NX + i;
    uint idx5 = jm1NX + im1;
    uint idx6 = jm1NX + ip1;
    uint idx7 = jp1NX + ip1;
    uint idx8 = jp1NX + im1;

    // Load streamed distributions
    float f0 = f_in[idx0];
    float f1 = f_in[1 * N + idx1];
    float f2 = f_in[2 * N + idx2];
    float f3 = f_in[3 * N + idx3];
    float f4 = f_in[4 * N + idx4];
    float f5 = f_in[5 * N + idx5];
    float f6 = f_in[6 * N + idx6];
    float f7 = f_in[7 * N + idx7];
    float f8 = f_in[8 * N + idx8];

    // Compute macroscopic moments
    float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    float ux = f1 - f3 + f5 - f6 - f7 + f8;
    float uy = f2 - f4 + f5 + f6 - f7 - f8;
    
    float inv_rho = 1.0f / rho;
    ux *= inv_rho;
    uy *= inv_rho;

    // Compute relaxation and common terms for equilibrium
    float omega = 1.0f / tau;
    float oma = 1.0f - omega;

    float w0_omega = omega * (4.0f / 9.0f);
    float w1_omega = omega * (1.0f / 9.0f);
    float w5_omega = omega * (1.0f / 36.0f);

    float rho_w0_omega = rho * w0_omega;
    float rho_w1_omega = rho * w1_omega;
    float rho_w5_omega = rho * w5_omega;

    float ux_sq = ux * ux;
    float uy_sq = uy * uy;
    float usq = ux_sq + uy_sq;
    float C = 1.0f - 1.5f * usq;

    // BGK Collision & writeback
    // K = 0
    f_out[idx0] = f0 * oma + rho_w0_omega * C;

    // K = 1, 3
    float term1 = C + 4.5f * ux_sq;
    float lin1 = 3.0f * ux;
    f_out[1 * N + idx0] = f1 * oma + rho_w1_omega * (term1 + lin1);
    f_out[3 * N + idx0] = f3 * oma + rho_w1_omega * (term1 - lin1);

    // K = 2, 4
    float term2 = C + 4.5f * uy_sq;
    float lin2 = 3.0f * uy;
    f_out[2 * N + idx0] = f2 * oma + rho_w1_omega * (term2 + lin2);
    f_out[4 * N + idx0] = f4 * oma + rho_w1_omega * (term2 - lin2);

    // K = 5, 7
    float cu5 = ux + uy;
    float term5 = C + 4.5f * cu5 * cu5;
    float lin5 = 3.0f * cu5;
    f_out[5 * N + idx0] = f5 * oma + rho_w5_omega * (term5 + lin5);
    f_out[7 * N + idx0] = f7 * oma + rho_w5_omega * (term5 - lin5);

    // K = 6, 8
    float cu6 = uy - ux;
    float term6 = C + 4.5f * cu6 * cu6;
    float lin6 = 3.0f * cu6;
    f_out[6 * N + idx0] = f6 * oma + rho_w5_omega * (term6 + lin6);
    f_out[8 * N + idx0] = f8 * oma + rho_w5_omega * (term6 - lin6);
}