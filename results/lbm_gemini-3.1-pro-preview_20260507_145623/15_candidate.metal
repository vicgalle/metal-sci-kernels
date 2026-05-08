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

    uint N = NX * NY;
    int inx = int(NX);
    int iny = int(NY);

    // Fast 1D periodic boundary offset calculation
    int im1 = (i == 0) ? (inx - 1) : -1;
    int ip1 = (i == NX - 1) ? (1 - inx) : 1;
    int jm1 = (j == 0) ? (inx * (iny - 1)) : -inx;
    int jp1 = (j == NY - 1) ? -(inx * (iny - 1)) : inx;

    // Precalculate base pointers for perfectly uniform strides
    uint i0 = j * NX + i;
    uint i1 = i0 + N;
    uint i2 = i1 + N;
    uint i3 = i2 + N;
    uint i4 = i3 + N;
    uint i5 = i4 + N;
    uint i6 = i5 + N;
    uint i7 = i6 + N;
    uint i8 = i7 + N;

    // Pull streaming reads (relying on 32-bit uint wrapping for negative offsets)
    float f0 = f_in[i0];
    float f1 = f_in[i1 + uint(im1)];
    float f2 = f_in[i2 + uint(jm1)];
    float f3 = f_in[i3 + uint(ip1)];
    float f4 = f_in[i4 + uint(jp1)];
    float f5 = f_in[i5 + uint(im1 + jm1)];
    float f6 = f_in[i6 + uint(ip1 + jm1)];
    float f7 = f_in[i7 + uint(ip1 + jp1)];
    float f8 = f_in[i8 + uint(im1 + jp1)];

    // Moments
    float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    float inv_rho = 1.0f / rho;
    
    float ux = (f1 - f3 + f5 - f6 - f7 + f8) * inv_rho;
    float uy = (f2 - f4 + f5 + f6 - f7 - f8) * inv_rho;

    float ux2 = ux * ux;
    float uy2 = uy * uy;
    float usq = ux2 + uy2;
    float inv_tau = 1.0f / tau;
    float term0 = 1.0f - 1.5f * usq;
    
    float w0_rho = rho * (4.0f / 9.0f);
    float w1_rho = rho * (1.0f / 9.0f);
    float w2_rho = rho * (1.0f / 36.0f);

    // BGK collision and writes (interleaved to minimize register lifespan)
    f_out[i0] = f0 - inv_tau * (f0 - w0_rho * term0);

    float ux3 = 3.0f * ux;
    float ux2_45 = 4.5f * ux2;
    f_out[i1] = f1 - inv_tau * (f1 - w1_rho * (term0 + ux3 + ux2_45));
    f_out[i3] = f3 - inv_tau * (f3 - w1_rho * (term0 - ux3 + ux2_45));

    float uy3 = 3.0f * uy;
    float uy2_45 = 4.5f * uy2;
    f_out[i2] = f2 - inv_tau * (f2 - w1_rho * (term0 + uy3 + uy2_45));
    f_out[i4] = f4 - inv_tau * (f4 - w1_rho * (term0 - uy3 + uy2_45));

    float cu5 = ux + uy;
    float cu5_2 = cu5 * cu5;
    float cu5_3 = 3.0f * cu5;
    float cu5_2_45 = 4.5f * cu5_2;
    f_out[i5] = f5 - inv_tau * (f5 - w2_rho * (term0 + cu5_3 + cu5_2_45));
    f_out[i7] = f7 - inv_tau * (f7 - w2_rho * (term0 - cu5_3 + cu5_2_45));

    float cu6 = -ux + uy;
    float cu6_2 = cu6 * cu6;
    float cu6_3 = 3.0f * cu6;
    float cu6_2_45 = 4.5f * cu6_2;
    f_out[i6] = f6 - inv_tau * (f6 - w2_rho * (term0 + cu6_3 + cu6_2_45));
    f_out[i8] = f8 - inv_tau * (f8 - w2_rho * (term0 - cu6_3 + cu6_2_45));
}