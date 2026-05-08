#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(256)]]
kernel void lbm_step(device const float *f_in   [[buffer(0)]],
                     device       float *f_out  [[buffer(1)]],
                     constant uint        &NX   [[buffer(2)]],
                     constant uint        &NY   [[buffer(3)]],
                     constant float       &tau  [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]]) {
    // Original thread coordinates
    uint i_in = gid.x;
    uint j_in = gid.y;
    if (i_in >= NX || j_in >= NY) return;

    // Remap the full NX*NY dispatch into fewer active threads.
    // Each active thread will handle a column of 4 cells along the Y-axis.
    uint tid = j_in * NX + i_in;
    
    const uint CELLS = 4;
    uint total_threads = NX * NY;
    uint active_threads = (total_threads + CELLS - 1) / CELLS;
    
    if (tid >= active_threads) return;

    uint N = total_threads;

    // 'i' varies fastest so memory access across the warp remains coalesced
    uint i = tid % NX;
    uint j_base = (tid / NX) * CELLS;

    uint im1 = (i == 0) ? NX - 1 : i - 1;
    uint ip1 = (i + 1 == NX) ? 0 : i + 1;

    uint j = j_base;
    if (j >= NY) return;

    uint jm1 = (j == 0) ? NY - 1 : j - 1;
    uint jp1 = (j + 1 == NY) ? 0 : j + 1;

    // Load initial 9 distributions for the first cell in the column
    float f2 = f_in[2 * N + jm1 * NX + i];
    float f5 = f_in[5 * N + jm1 * NX + im1];
    float f6 = f_in[6 * N + jm1 * NX + ip1];

    float f0 = f_in[0 * N + j * NX + i];
    float f1 = f_in[1 * N + j * NX + im1];
    float f3 = f_in[3 * N + j * NX + ip1];

    uint jp1_NX = jp1 * NX;
    float f4 = f_in[4 * N + jp1_NX + i];
    float f8 = f_in[8 * N + jp1_NX + im1];
    float f7 = f_in[7 * N + jp1_NX + ip1];

    float inv_tau_val = 1.0f / tau;
    float om_inv_tau = 1.0f - inv_tau_val;

    for (uint k = 0; k < CELLS; ++k) {
        uint current_j = j_base + k;
        uint idx = current_j * NX + i;

        // Moments computation
        float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
        float inv_rho = 1.0f / rho;
        float ux = (f1 - f3 + f5 - f6 - f7 + f8) * inv_rho;
        float uy = (f2 - f4 + f5 + f6 - f7 - f8) * inv_rho;

        // BGK collision preparation - optimized algebraic grouping
        float sq_x = ux * ux;
        float sq_y = uy * uy;
        float om_usq15 = 1.0f - 1.5f * (sq_x + sq_y);

        float T = inv_tau_val * rho;
        float T_w0 = T * (4.0f / 9.0f);
        float T_w1 = T * (1.0f / 9.0f);
        float T_w5 = T * (1.0f / 36.0f);

        float term_x = om_usq15 + 4.5f * sq_x;
        float lin_x = 3.0f * ux;
        float feq1_part = term_x + lin_x;
        float feq3_part = term_x - lin_x;

        float term_y = om_usq15 + 4.5f * sq_y;
        float lin_y = 3.0f * uy;
        float feq2_part = term_y + lin_y;
        float feq4_part = term_y - lin_y;

        float cu5 = ux + uy;
        float term_5 = om_usq15 + 4.5f * (cu5 * cu5);
        float lin_5 = 3.0f * cu5;
        float feq5_part = term_5 + lin_5;
        float feq7_part = term_5 - lin_5;

        float cu6 = -ux + uy;
        float term_6 = om_usq15 + 4.5f * (cu6 * cu6);
        float lin_6 = 3.0f * cu6;
        float feq6_part = term_6 + lin_6;
        float feq8_part = term_6 - lin_6;

        // Write outputs
        f_out[idx]         = fma(f0, om_inv_tau, T_w0 * om_usq15);
        f_out[N + idx]     = fma(f1, om_inv_tau, T_w1 * feq1_part);
        f_out[2 * N + idx] = fma(f2, om_inv_tau, T_w1 * feq2_part);
        f_out[3 * N + idx] = fma(f3, om_inv_tau, T_w1 * feq3_part);
        f_out[4 * N + idx] = fma(f4, om_inv_tau, T_w1 * feq4_part);
        f_out[5 * N + idx] = fma(f5, om_inv_tau, T_w5 * feq5_part);
        f_out[6 * N + idx] = fma(f6, om_inv_tau, T_w5 * feq6_part);
        f_out[7 * N + idx] = fma(f7, om_inv_tau, T_w5 * feq7_part);
        f_out[8 * N + idx] = fma(f8, om_inv_tau, T_w5 * feq8_part);

        // Slide the 6 overlapping distributions down the column and read the 3 new ones
        if (k < CELLS - 1) {
            uint next_j = current_j + 1;
            if (next_j >= NY) break;

            f2 = f0; f5 = f1; f6 = f3;
            f0 = f4; f1 = f8; f3 = f7;

            uint next_jp1 = (next_j + 1 == NY) ? 0 : next_j + 1;
            uint next_idx = next_jp1 * NX;
            f4 = f_in[4 * N + next_idx + i];
            f8 = f_in[8 * N + next_idx + im1];
            f7 = f_in[7 * N + next_idx + ip1];
        }
    }
}