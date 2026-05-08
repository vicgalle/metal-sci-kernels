#include <metal_stdlib>
using namespace metal;

inline float feq_term(float W, float rho, float cu, float c1) {
    return W * rho * (c1 + 3.0f * cu + 4.5f * cu * cu);
}

kernel void lbm_step(device const float *f_in   [[buffer(0)]],
                     device       float *f_out  [[buffer(1)]],
                     constant uint        &NX   [[buffer(2)]],
                     constant uint        &NY   [[buffer(3)]],
                     constant float       &tau  [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint j = gid.y;
    if (i >= NX || j >= NY) return;

    uint nx = NX;
    uint ny = NY;
    uint N  = nx * ny;

    uint ip1 = (i + 1u == nx) ? 0u : (i + 1u);
    uint im1 = (i == 0u) ? (nx - 1u) : (i - 1u);
    uint jp1 = (j + 1u == ny) ? 0u : (j + 1u);
    uint jm1 = (j == 0u) ? (ny - 1u) : (j - 1u);

    uint row    = j   * nx;
    uint rowU   = jm1 * nx;
    uint rowD   = jp1 * nx;

    uint b0 = 0u;
    uint b1 = N;
    uint b2 = 2u * N;
    uint b3 = 3u * N;
    uint b4 = 4u * N;
    uint b5 = 5u * N;
    uint b6 = 6u * N;
    uint b7 = 7u * N;
    uint b8 = 8u * N;

    float f0 = f_in[b0 + row  + i  ];
    float f1 = f_in[b1 + row  + im1];
    float f2 = f_in[b2 + rowU + i  ];
    float f3 = f_in[b3 + row  + ip1];
    float f4 = f_in[b4 + rowD + i  ];
    float f5 = f_in[b5 + rowU + im1];
    float f6 = f_in[b6 + rowU + ip1];
    float f7 = f_in[b7 + rowD + ip1];
    float f8 = f_in[b8 + rowD + im1];

    float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    float jx  = (f1 - f3) + (f5 - f6) + (f8 - f7);
    float jy  = (f2 - f4) + (f5 + f6) - (f7 + f8);
    float inv_rho = 1.0f / rho;
    float ux = jx * inv_rho;
    float uy = jy * inv_rho;

    float usq = ux * ux + uy * uy;
    float omega = 1.0f / tau;
    float one_minus_omega = 1.0f - omega;
    float c1 = 1.0f - 1.5f * usq;

    const float W0 = 4.0f / 9.0f;
    const float W1 = 1.0f / 9.0f;
    const float W5 = 1.0f / 36.0f;

    // Precompute rho-scaled weights and omega-scaled weights to reduce mults.
    float wr0 = omega * W0 * rho;
    float wr1 = omega * W1 * rho;
    float wr5 = omega * W5 * rho;

    uint idx = row + i;

    // k=0: cu = 0
    f_out[b0 + idx] = one_minus_omega * f0 + wr0 * c1;

    // k=1: cu = ux
    {
        float cu = ux;
        f_out[b1 + idx] = one_minus_omega * f1 + wr1 * (c1 + 3.0f * cu + 4.5f * cu * cu);
    }
    // k=2: cu = uy
    {
        float cu = uy;
        f_out[b2 + idx] = one_minus_omega * f2 + wr1 * (c1 + 3.0f * cu + 4.5f * cu * cu);
    }
    // k=3: cu = -ux
    {
        float cu = -ux;
        f_out[b3 + idx] = one_minus_omega * f3 + wr1 * (c1 + 3.0f * cu + 4.5f * cu * cu);
    }
    // k=4: cu = -uy
    {
        float cu = -uy;
        f_out[b4 + idx] = one_minus_omega * f4 + wr1 * (c1 + 3.0f * cu + 4.5f * cu * cu);
    }
    // k=5: cu = ux + uy
    {
        float cu = ux + uy;
        f_out[b5 + idx] = one_minus_omega * f5 + wr5 * (c1 + 3.0f * cu + 4.5f * cu * cu);
    }
    // k=6: cu = -ux + uy
    {
        float cu = -ux + uy;
        f_out[b6 + idx] = one_minus_omega * f6 + wr5 * (c1 + 3.0f * cu + 4.5f * cu * cu);
    }
    // k=7: cu = -ux - uy
    {
        float cu = -ux - uy;
        f_out[b7 + idx] = one_minus_omega * f7 + wr5 * (c1 + 3.0f * cu + 4.5f * cu * cu);
    }
    // k=8: cu = ux - uy
    {
        float cu = ux - uy;
        f_out[b8 + idx] = one_minus_omega * f8 + wr5 * (c1 + 3.0f * cu + 4.5f * cu * cu);
    }
}