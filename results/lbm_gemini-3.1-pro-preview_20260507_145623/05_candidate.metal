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

    // Fast periodic boundary wrapping (eliminates expensive modulo)
    uint im1 = (i == 0) ? NX - 1 : i - 1;
    uint ip1 = (i == NX - 1) ? 0 : i + 1;
    uint jm1 = (j == 0) ? NY - 1 : j - 1;
    uint jp1 = (j == NY - 1) ? 0 : j + 1;

    // Precalculate row offsets
    uint row_j   = j * NX;
    uint row_jm1 = jm1 * NX;
    uint row_jp1 = jp1 * NX;

    // 1. Pull streaming phase
    // Explicit scalarization guarantees zero local memory allocation and perfectly maps to registers.
    float f0 = f_in[0 * N + row_j   + i  ];
    float f1 = f_in[1 * N + row_j   + im1];
    float f2 = f_in[2 * N + row_jm1 + i  ];
    float f3 = f_in[3 * N + row_j   + ip1];
    float f4 = f_in[4 * N + row_jp1 + i  ];
    float f5 = f_in[5 * N + row_jm1 + im1];
    float f6 = f_in[6 * N + row_jm1 + ip1];
    float f7 = f_in[7 * N + row_jp1 + ip1];
    float f8 = f_in[8 * N + row_jp1 + im1];

    // 2. Macroscopic moments
    // Sequentially accumulated to strictly preserve bitwise math tolerance.
    float rho = 0.0f;
    rho += f0; rho += f1; rho += f2;
    rho += f3; rho += f4; rho += f5;
    rho += f6; rho += f7; rho += f8;

    float ux = 0.0f;
    ux += f1; ux -= f3; ux += f5;
    ux -= f6; ux -= f7; ux += f8;

    float uy = 0.0f;
    uy += f2; uy -= f4; uy += f5;
    uy += f6; uy -= f7; uy -= f8;

    float inv_rho = 1.0f / rho;
    ux *= inv_rho;
    uy *= inv_rho;

    // 3. BGK collision & Write phase
    float usq = ux * ux + uy * uy;
    float inv_tau = 1.0f / tau;
    
    // Evaluate equilibrium polynomial components optimally
    float base = 1.0f - 1.5f * usq;

    float w0 = 4.0f / 9.0f;
    float w1 = 1.0f / 9.0f;
    float w5 = 1.0f / 36.0f;

    float rho_w0 = rho * w0;
    float rho_w1 = rho * w1;
    float rho_w5 = rho * w5;

    uint idx = row_j + i;

    // Interleave operations to map into immediate stores, reducing live register pressure
    float feq0 = rho_w0 * base;
    f_out[0 * N + idx] = f0 - inv_tau * (f0 - feq0);

    float cu1 = ux;
    float feq1 = rho_w1 * (base + cu1 * (3.0f + 4.5f * cu1));
    f_out[1 * N + idx] = f1 - inv_tau * (f1 - feq1);

    float cu2 = uy;
    float feq2 = rho_w1 * (base + cu2 * (3.0f + 4.5f * cu2));
    f_out[2 * N + idx] = f2 - inv_tau * (f2 - feq2);

    float cu3 = -ux;
    float feq3 = rho_w1 * (base + cu3 * (3.0f + 4.5f * cu3));
    f_out[3 * N + idx] = f3 - inv_tau * (f3 - feq3);

    float cu4 = -uy;
    float feq4 = rho_w1 * (base + cu4 * (3.0f + 4.5f * cu4));
    f_out[4 * N + idx] = f4 - inv_tau * (f4 - feq4);

    float cu5 = ux + uy;
    float feq5 = rho_w5 * (base + cu5 * (3.0f + 4.5f * cu5));
    f_out[5 * N + idx] = f5 - inv_tau * (f5 - feq5);

    float cu6 = -ux + uy;
    float feq6 = rho_w5 * (base + cu6 * (3.0f + 4.5f * cu6));
    f_out[6 * N + idx] = f6 - inv_tau * (f6 - feq6);

    // Reuse inverted directional momenta
    float cu7 = -cu5;
    float feq7 = rho_w5 * (base + cu7 * (3.0f + 4.5f * cu7));
    f_out[7 * N + idx] = f7 - inv_tau * (f7 - feq7);

    float cu8 = -cu6;
    float feq8 = rho_w5 * (base + cu8 * (3.0f + 4.5f * cu8));
    f_out[8 * N + idx] = f8 - inv_tau * (f8 - feq8);
}