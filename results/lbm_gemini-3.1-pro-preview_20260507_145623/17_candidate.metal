#include <metal_stdlib>
using namespace metal;

kernel void lbm_step(device const float *f_in   [[buffer(0)]],
                     device       float *f_out  [[buffer(1)]],
                     constant uint        &NX   [[buffer(2)]],
                     constant uint        &NY   [[buffer(3)]],
                     constant float       &tau  [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint j = gid.y;
    
    if (i >= NX || j >= NY) return;

    int inx = int(NX);
    int iny = int(NY);
    uint N = NX * NY;
    uint idx = j * NX + i;

    // Fast 1D periodic boundary offset calculation
    int im1 = (i == 0) ? (inx - 1) : -1;
    int ip1 = (i == NX - 1) ? (1 - inx) : 1;
    int jm1 = (j == 0) ? (inx * (iny - 1)) : -inx;
    int jp1 = (j == NY - 1) ? -(inx * (iny - 1)) : inx;

    // 1) Pull streaming: issue 9 independent loads
    float f0 = f_in[idx];
    float f1 = f_in[1 * N + idx + im1];
    float f2 = f_in[2 * N + idx + jm1];
    float f3 = f_in[3 * N + idx + ip1];
    float f4 = f_in[4 * N + idx + jp1];
    float f5 = f_in[5 * N + idx + im1 + jm1];
    float f6 = f_in[6 * N + idx + ip1 + jm1];
    float f7 = f_in[7 * N + idx + ip1 + jp1];
    float f8 = f_in[8 * N + idx + im1 + jp1];

    // 2) Moments
    float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;

    float ux = (f1 + f5 + f8) - (f3 + f6 + f7);
    float uy = (f2 + f5 + f6) - (f4 + f7 + f8);

    float inv_rho = 1.0f / rho;
    float u = ux * inv_rho;
    float v = uy * inv_rho;

    float usq = u * u + v * v;
    float term0 = 1.0f - 1.5f * usq;

    float r_w0 = rho * (4.0f / 9.0f);
    float r_w1 = rho * (1.0f / 9.0f);
    float r_w5 = rho * (1.0f / 36.0f);

    float inv_tau = 1.0f / tau;

    // 3) BGK Collision explicitly unrolled (maps to tight FMAs)
    float feq0 = r_w0 * term0;
    
    float cu1 = u;
    float feq1 = r_w1 * (term0 + cu1 * (3.0f + 4.5f * cu1));
    
    float cu2 = v;
    float feq2 = r_w1 * (term0 + cu2 * (3.0f + 4.5f * cu2));
    
    float cu3 = -u;
    float feq3 = r_w1 * (term0 + cu3 * (3.0f + 4.5f * cu3));
    
    float cu4 = -v;
    float feq4 = r_w1 * (term0 + cu4 * (3.0f + 4.5f * cu4));
    
    float cu5 = u + v;
    float feq5 = r_w5 * (term0 + cu5 * (3.0f + 4.5f * cu5));
    
    float cu6 = -u + v;
    float feq6 = r_w5 * (term0 + cu6 * (3.0f + 4.5f * cu6));
    
    float cu7 = -cu5; // saves an ADD
    float feq7 = r_w5 * (term0 + cu7 * (3.0f + 4.5f * cu7));
    
    float cu8 = -cu6; // saves an ADD
    float feq8 = r_w5 * (term0 + cu8 * (3.0f + 4.5f * cu8));

    // Write back
    f_out[idx]         = f0 - inv_tau * (f0 - feq0);
    f_out[1 * N + idx] = f1 - inv_tau * (f1 - feq1);
    f_out[2 * N + idx] = f2 - inv_tau * (f2 - feq2);
    f_out[3 * N + idx] = f3 - inv_tau * (f3 - feq3);
    f_out[4 * N + idx] = f4 - inv_tau * (f4 - feq4);
    f_out[5 * N + idx] = f5 - inv_tau * (f5 - feq5);
    f_out[6 * N + idx] = f6 - inv_tau * (f6 - feq6);
    f_out[7 * N + idx] = f7 - inv_tau * (f7 - feq7);
    f_out[8 * N + idx] = f8 - inv_tau * (f8 - feq8);
}