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
    
    // Fast periodic boundary wrapping, avoiding modulo or branching
    uint ip1 = (i + 1 == NX) ? 0 : i + 1;
    uint im1 = (i == 0) ? NX - 1 : i - 1;
    
    // Y-boundary linear indices evaluated directly
    uint jm1_NX = (j == 0) ? (N - NX) : (j_NX - NX);
    uint jp1_NX = (j + 1 == NY) ? 0 : (j_NX + NX);

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

    // 2. Moments computation (tree-reduced for maximal ILP)
    float rho = (f0 + f1 + f2) + (f3 + f4 + f5) + (f6 + f7 + f8);
    float inv_rho = 1.0f / rho;
    
    float ux = ((f1 + f5 + f8) - (f3 + f6 + f7)) * inv_rho;
    float uy = ((f2 + f5 + f6) - (f4 + f7 + f8)) * inv_rho;

    // 3. BGK collision preparation via fully fused Horner polynomials
    float usq = fma(ux, ux, uy * uy); // ux*ux + uy*uy
    float om_usq15 = 1.0f - 1.5f * usq;

    float inv_tau_val = 1.0f / tau;
    float om_inv_tau = 1.0f - inv_tau_val;

    // Pre-bake inverse tau into the equilibrium weights
    float rho_w0_it = rho * (inv_tau_val * (4.0f / 9.0f));
    float rho_w1_it = rho * (inv_tau_val * (1.0f / 9.0f));
    float rho_w5_it = rho * (inv_tau_val * (1.0f / 36.0f));

    // Shared coefficients for Horner polynomial evaluation
    float r1_45 = rho_w1_it * 4.5f;
    float r1_30 = rho_w1_it * 3.0f;
    float r1_om = rho_w1_it * om_usq15;

    float r5_45 = rho_w5_it * 4.5f;
    float r5_30 = rho_w5_it * 3.0f;
    float r5_om = rho_w5_it * om_usq15;

    uint idx = j_NX + i;

    // Evaluate BGK and write out (exactly 3 FMAs per direction)
    f_out[idx]         = fma(f0, om_inv_tau, rho_w0_it * om_usq15);
    f_out[N + idx]     = fma(f1, om_inv_tau, fma( ux, fma( ux, r1_45, r1_30), r1_om));
    f_out[2 * N + idx] = fma(f2, om_inv_tau, fma( uy, fma( uy, r1_45, r1_30), r1_om));
    f_out[3 * N + idx] = fma(f3, om_inv_tau, fma(-ux, fma(-ux, r1_45, r1_30), r1_om));
    f_out[4 * N + idx] = fma(f4, om_inv_tau, fma(-uy, fma(-uy, r1_45, r1_30), r1_om));

    float cu5 = ux + uy;
    f_out[5 * N + idx] = fma(f5, om_inv_tau, fma(cu5, fma(cu5, r5_45, r5_30), r5_om));

    float cu6 = -ux + uy;
    f_out[6 * N + idx] = fma(f6, om_inv_tau, fma(cu6, fma(cu6, r5_45, r5_30), r5_om));

    float cu7 = -cu5;
    f_out[7 * N + idx] = fma(f7, om_inv_tau, fma(cu7, fma(cu7, r5_45, r5_30), r5_om));

    float cu8 = -cu6;
    f_out[8 * N + idx] = fma(f8, om_inv_tau, fma(cu8, fma(cu8, r5_45, r5_30), r5_om));
}