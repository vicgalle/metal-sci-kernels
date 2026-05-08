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

    // Fast periodic boundary wrapping
    uint ip1 = (i == NX - 1) ? 0 : i + 1;
    uint im1 = (i == 0) ? NX - 1 : i - 1;
    uint jp1 = (j == NY - 1) ? 0 : j + 1;
    uint jm1 = (j == 0) ? NY - 1 : j - 1;

    uint N = NX * NY;
    uint j_NX = j * NX;
    uint jm1_NX = jm1 * NX;
    uint jp1_NX = jp1 * NX;

    // Calculate memory indices for the 9 streaming directions
    uint idx0 = j_NX + i;
    uint idx1 = j_NX + im1;
    uint idx2 = jm1_NX + i;
    uint idx3 = j_NX + ip1;
    uint idx4 = jp1_NX + i;
    uint idx5 = jm1_NX + im1;
    uint idx6 = jm1_NX + ip1;
    uint idx7 = jp1_NX + ip1;
    uint idx8 = jp1_NX + im1;

    // 1. Pull streaming
    float f0 = f_in[idx0];
    float f1 = f_in[N + idx1];
    float f2 = f_in[2 * N + idx2];
    float f3 = f_in[3 * N + idx3];
    float f4 = f_in[4 * N + idx4];
    float f5 = f_in[5 * N + idx5];
    float f6 = f_in[6 * N + idx6];
    float f7 = f_in[7 * N + idx7];
    float f8 = f_in[8 * N + idx8];

    // 2. Moments computation
    float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    float inv_rho = 1.0f / rho;
    
    float ux = ((f1 + f5 + f8) - (f3 + f6 + f7)) * inv_rho;
    float uy = ((f2 + f5 + f6) - (f4 + f7 + f8)) * inv_rho;

    // 3. BGK collision preparation - optimized via FMA and symmetry
    float inv_tau_val = 1.0f / tau;
    float om_inv_tau  = 1.0f - inv_tau_val;

    // Pre-multiply 1/tau into the weighted density to save operations
    float rho_w0_it = rho * (inv_tau_val * (4.0f / 9.0f));
    float rho_w1_it = rho * (inv_tau_val * (1.0f / 9.0f));
    float rho_w5_it = rho * (inv_tau_val * (1.0f / 36.0f));

    float ux_sq = ux * ux;
    float uy_sq = uy * uy;
    float om_usq15 = 1.0f - 1.5f * (ux_sq + uy_sq);

    float cu5 = ux + uy;
    float cu6 = -ux + uy;

    // Compute shared quadratic terms for symmetric directions
    float term1  = fma(ux_sq, 4.5f, om_usq15);
    float term2  = fma(uy_sq, 4.5f, om_usq15);
    float term57 = fma(cu5 * cu5, 4.5f, om_usq15);
    float term68 = fma(cu6 * cu6, 4.5f, om_usq15);

    // Collision writes back to identical flat coordinate indices via optimized FMAs
    f_out[idx0]         = fma(f0, om_inv_tau, rho_w0_it * om_usq15);
    f_out[N + idx0]     = fma(f1, om_inv_tau, rho_w1_it * fma( ux, 3.0f, term1));
    f_out[2 * N + idx0] = fma(f2, om_inv_tau, rho_w1_it * fma( uy, 3.0f, term2));
    f_out[3 * N + idx0] = fma(f3, om_inv_tau, rho_w1_it * fma(-ux, 3.0f, term1));
    f_out[4 * N + idx0] = fma(f4, om_inv_tau, rho_w1_it * fma(-uy, 3.0f, term2));
    f_out[5 * N + idx0] = fma(f5, om_inv_tau, rho_w5_it * fma( cu5, 3.0f, term57));
    f_out[6 * N + idx0] = fma(f6, om_inv_tau, rho_w5_it * fma( cu6, 3.0f, term68));
    f_out[7 * N + idx0] = fma(f7, om_inv_tau, rho_w5_it * fma(-cu5, 3.0f, term57));
    f_out[8 * N + idx0] = fma(f8, om_inv_tau, rho_w5_it * fma(-cu6, 3.0f, term68));
}