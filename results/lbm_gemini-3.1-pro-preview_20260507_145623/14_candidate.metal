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
    uint nx = NX;
    uint ny = NY;
    
    if (i >= nx || j >= ny) return;

    int inx = int(nx);
    int iny = int(ny);
    int N = inx * iny;
    int idx = int(j * nx + i);

    int im1_val = inx - 1;
    int jm1_val = N - inx;
    
    int im1 = (i == 0) ? im1_val : -1;
    int ip1 = (i == nx - 1) ? -im1_val : 1;
    int jm1 = (j == 0) ? jm1_val : -inx;
    int jp1 = (j == ny - 1) ? -jm1_val : inx;

    // Pull streaming reads
    float f0 = f_in[idx];
    float f1 = f_in[1 * N + idx + im1];
    float f2 = f_in[2 * N + idx + jm1];
    float f3 = f_in[3 * N + idx + ip1];
    float f4 = f_in[4 * N + idx + jp1];
    float f5 = f_in[5 * N + idx + im1 + jm1];
    float f6 = f_in[6 * N + idx + ip1 + jm1];
    float f7 = f_in[7 * N + idx + ip1 + jp1];
    float f8 = f_in[8 * N + idx + im1 + jp1];

    // Symmetries for moments
    float f1_f3 = f1 - f3;
    float f5_f7 = f5 - f7;
    float f8_f6 = f8 - f6;
    float f1_p_f3 = f1 + f3;
    float f2_p_f4 = f2 + f4;
    float f5_p_f7 = f5 + f7;
    float f6_p_f8 = f6 + f8;

    float rho = f0 + f1_p_f3 + f2_p_f4 + f5_p_f7 + f6_p_f8;
    float inv_rho = 1.0f / rho;
    
    float ux = (f1_f3 + f5_f7 + f8_f6) * inv_rho;
    float uy = ((f2 - f4) + f5_f7 - f8_f6) * inv_rho;

    // Precompute constants for BGK
    float cu1_sq = ux * ux;
    float cu2_sq = uy * uy;
    float usq = cu1_sq + cu2_sq;
    float om_usq15 = 1.0f - 1.5f * usq;

    float w9_rho = rho * (1.0f / 9.0f);
    float w36_rho = rho * (1.0f / 36.0f);

    float ux3 = 3.0f * ux;
    float uy3 = 3.0f * uy;

    float base1 = om_usq15 + 4.5f * cu1_sq;
    float feq1 = w9_rho * (base1 + ux3);
    float feq3 = w9_rho * (base1 - ux3);

    float base2 = om_usq15 + 4.5f * cu2_sq;
    float feq2 = w9_rho * (base2 + uy3);
    float feq4 = w9_rho * (base2 - uy3);

    float ux_uy_9 = 9.0f * (ux * uy);
    float base56 = 1.0f + 3.0f * usq;
    float base5 = base56 + ux_uy_9;
    float base6 = base56 - ux_uy_9;

    float uxy3_plus = ux3 + uy3;
    float uxy3_minus = uy3 - ux3;

    float feq5 = w36_rho * (base5 + uxy3_plus);
    float feq7 = w36_rho * (base5 - uxy3_plus);

    float feq6 = w36_rho * (base6 + uxy3_minus);
    float feq8 = w36_rho * (base6 - uxy3_minus);

    float feq0 = rho * (4.0f / 9.0f) * om_usq15;

    float inv_tau = 1.0f / tau;

    // Writes
    f_out[idx] = f0 + inv_tau * (feq0 - f0);
    f_out[1 * N + idx] = f1 + inv_tau * (feq1 - f1);
    f_out[2 * N + idx] = f2 + inv_tau * (feq2 - f2);
    f_out[3 * N + idx] = f3 + inv_tau * (feq3 - f3);
    f_out[4 * N + idx] = f4 + inv_tau * (feq4 - f4);
    f_out[5 * N + idx] = f5 + inv_tau * (feq5 - f5);
    f_out[6 * N + idx] = f6 + inv_tau * (feq6 - f6);
    f_out[7 * N + idx] = f7 + inv_tau * (feq7 - f7);
    f_out[8 * N + idx] = f8 + inv_tau * (feq8 - f8);
}