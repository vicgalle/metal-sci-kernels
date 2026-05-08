#include <metal_stdlib>
using namespace metal;

kernel void lbm_step(device const float *f_in   [[buffer(0)]],
                     device       float *f_out  [[buffer(1)]],
                     constant uint        &NX   [[buffer(2)]],
                     constant uint        &NY   [[buffer(3)]],
                     constant float       &tau  [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]]) {
    const uint nx = NX;
    const uint ny = NY;
    
    if (gid.x >= nx || gid.y >= ny) return;

    const uint i = gid.x;
    const uint j = gid.y;
    const uint N = nx * ny;

    // Fast periodic boundary wrapping using conditionally selected constants
    const uint im1 = (i > 0) ? i - 1 : nx - 1;
    const uint ip1 = (i == nx - 1) ? 0 : i + 1;
    const uint jm1 = (j > 0) ? j - 1 : ny - 1;
    const uint jp1 = (j == ny - 1) ? 0 : j + 1;

    const uint y_j   = j * nx;
    const uint y_jm1 = jm1 * nx;
    const uint y_jp1 = jp1 * nx;

    const uint idx = y_j + i;

    // Fully scalarized pull-streaming memory fetches map directly to registers
    // and naturally benefit from adjacent 32-thread coalesced vector loads.
    float f0 = f_in[idx];
    float f1 = f_in[1 * N + y_j + im1];
    float f2 = f_in[2 * N + y_jm1 + i];
    float f3 = f_in[3 * N + y_j + ip1];
    float f4 = f_in[4 * N + y_jp1 + i];
    float f5 = f_in[5 * N + y_jm1 + im1];
    float f6 = f_in[6 * N + y_jm1 + ip1];
    float f7 = f_in[7 * N + y_jp1 + ip1];
    float f8 = f_in[8 * N + y_jp1 + im1];

    // Compute macroscopic moments
    float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    float inv_rho = 1.0f / rho;
    
    float ux = (f1 - f3 + f5 - f6 - f7 + f8) * inv_rho;
    float uy = (f2 - f4 + f5 + f6 - f7 - f8) * inv_rho;

    // Precompute constants and base equilibrium terms
    float usq_term = 1.0f - 1.5f * (ux * ux + uy * uy);
    
    float omega = 1.0f / tau;
    float om_omega = 1.0f - omega;
    float omega_rho = omega * rho;

    // Distribute omega into the polynomial coefficients early
    float w0_rho = omega_rho * (4.0f / 9.0f);
    float w1_rho = omega_rho * (1.0f / 9.0f);
    float w5_rho = omega_rho * (1.0f / 36.0f);

    float w0_usq = w0_rho * usq_term;
    float w1_usq = w1_rho * usq_term;
    float w5_usq = w5_rho * usq_term;

    float w1_rho_3  = w1_rho * 3.0f;
    float w1_rho_45 = w1_rho * 4.5f;
    float w5_rho_3  = w5_rho * 3.0f;
    float w5_rho_45 = w5_rho * 4.5f;

    // Axis directions (W[1..4])
    // Opposite directions perfectly share the quadratic component.
    float ux_sq = ux * ux;
    float uy_sq = uy * uy;
    
    float t1_base = fma(w1_rho_45, ux_sq, w1_usq);
    float t2_base = fma(w1_rho_45, uy_sq, w1_usq);

    float w1_ux_3 = ux * w1_rho_3;
    float w1_uy_3 = uy * w1_rho_3;

    float feq1 = t1_base + w1_ux_3;
    float feq3 = t1_base - w1_ux_3;
    float feq2 = t2_base + w1_uy_3;
    float feq4 = t2_base - w1_uy_3;

    // Diagonal directions (W[5..8])
    float cu5 = ux + uy;
    float cu8 = ux - uy;
    
    float cu5_sq = cu5 * cu5;
    float cu8_sq = cu8 * cu8;
    
    float t5_base = fma(w5_rho_45, cu5_sq, w5_usq);
    float t8_base = fma(w5_rho_45, cu8_sq, w5_usq);
    
    float w5_cu5_3 = cu5 * w5_rho_3;
    float w5_cu8_3 = cu8 * w5_rho_3;
    
    // Note: cu6 = -cu8 and cu7 = -cu5, so t8_base handles feq6, and t5_base handles feq7.
    float feq5 = t5_base + w5_cu5_3;
    float feq7 = t5_base - w5_cu5_3;
    float feq8 = t8_base + w5_cu8_3;
    float feq6 = t8_base - w5_cu8_3;

    // Fused BGK scatter: f_out = f_in * (1 - omega) + omega_feq
    f_out[idx]         = fma(f0, om_omega, w0_usq);
    f_out[1 * N + idx] = fma(f1, om_omega, feq1);
    f_out[2 * N + idx] = fma(f2, om_omega, feq2);
    f_out[3 * N + idx] = fma(f3, om_omega, feq3);
    f_out[4 * N + idx] = fma(f4, om_omega, feq4);
    f_out[5 * N + idx] = fma(f5, om_omega, feq5);
    f_out[6 * N + idx] = fma(f6, om_omega, feq6);
    f_out[7 * N + idx] = fma(f7, om_omega, feq7);
    f_out[8 * N + idx] = fma(f8, om_omega, feq8);
}