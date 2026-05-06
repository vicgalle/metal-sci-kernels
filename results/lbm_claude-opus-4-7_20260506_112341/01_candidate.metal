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

    uint N  = NX * NY;

    // Precompute wrapped neighbor indices (branchless modulo for ±1).
    uint im1 = (i == 0)      ? (NX - 1) : (i - 1);
    uint ip1 = (i + 1 == NX) ? 0u       : (i + 1);
    uint jm1 = (j == 0)      ? (NY - 1) : (j - 1);
    uint jp1 = (j + 1 == NY) ? 0u       : (j + 1);

    uint row    = j   * NX;
    uint row_m  = jm1 * NX;
    uint row_p  = jp1 * NX;

    // Pull streaming: source = (i - CX[k], j - CY[k]) mod (NX, NY).
    // CX = {0, 1, 0,-1, 0, 1,-1,-1, 1}
    // CY = {0, 0, 1, 0,-1, 1, 1,-1,-1}
    float f0 = f_in[0u * N + row   + i  ];
    float f1 = f_in[1u * N + row   + im1];
    float f2 = f_in[2u * N + row_m + i  ];
    float f3 = f_in[3u * N + row   + ip1];
    float f4 = f_in[4u * N + row_p + i  ];
    float f5 = f_in[5u * N + row_m + im1];
    float f6 = f_in[6u * N + row_m + ip1];
    float f7 = f_in[7u * N + row_p + ip1];
    float f8 = f_in[8u * N + row_p + im1];

    // Moments.
    float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    float inv_rho = 1.0f / rho;
    float ux = ((f1 - f3) + (f5 - f6) - (f7 - f8)) * inv_rho;
    float uy = ((f2 - f4) + (f5 + f6) - (f7 + f8)) * inv_rho;

    float usq     = ux * ux + uy * uy;
    float inv_tau = 1.0f / tau;
    float omega   = inv_tau;
    float one_m_w = 1.0f - omega;

    // Constants.
    const float W0 = 4.0f / 9.0f;
    const float WS = 1.0f / 9.0f;
    const float WD = 1.0f / 36.0f;

    float c1_5_usq = 1.5f * usq;

    // f_eq[k] = W[k] * rho * (1 + 3 cu + 4.5 cu^2 - 1.5 usq)
    // Update: f_out = (1 - omega) * f + omega * feq
    uint idx = row + i;

    // k=0: cu=0
    {
        float feq = W0 * rho * (1.0f - c1_5_usq);
        f_out[0u * N + idx] = one_m_w * f0 + omega * feq;
    }
    // k=1: c=( 1, 0), cu = ux
    {
        float cu = ux;
        float feq = WS * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - c1_5_usq);
        f_out[1u * N + idx] = one_m_w * f1 + omega * feq;
    }
    // k=2: c=( 0, 1), cu = uy
    {
        float cu = uy;
        float feq = WS * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - c1_5_usq);
        f_out[2u * N + idx] = one_m_w * f2 + omega * feq;
    }
    // k=3: c=(-1, 0), cu = -ux
    {
        float cu = -ux;
        float feq = WS * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - c1_5_usq);
        f_out[3u * N + idx] = one_m_w * f3 + omega * feq;
    }
    // k=4: c=( 0,-1), cu = -uy
    {
        float cu = -uy;
        float feq = WS * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - c1_5_usq);
        f_out[4u * N + idx] = one_m_w * f4 + omega * feq;
    }
    // k=5: c=( 1, 1), cu = ux + uy
    {
        float cu = ux + uy;
        float feq = WD * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - c1_5_usq);
        f_out[5u * N + idx] = one_m_w * f5 + omega * feq;
    }
    // k=6: c=(-1, 1), cu = -ux + uy
    {
        float cu = -ux + uy;
        float feq = WD * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - c1_5_usq);
        f_out[6u * N + idx] = one_m_w * f6 + omega * feq;
    }
    // k=7: c=(-1,-1), cu = -ux - uy
    {
        float cu = -ux - uy;
        float feq = WD * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - c1_5_usq);
        f_out[7u * N + idx] = one_m_w * f7 + omega * feq;
    }
    // k=8: c=( 1,-1), cu = ux - uy
    {
        float cu = ux - uy;
        float feq = WD * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - c1_5_usq);
        f_out[8u * N + idx] = one_m_w * f8 + omega * feq;
    }
}