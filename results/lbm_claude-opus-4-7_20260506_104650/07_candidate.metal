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

    uint im1 = (i == 0u)       ? (NX - 1u) : (i - 1u);
    uint ip1 = (i == NX - 1u)  ? 0u        : (i + 1u);
    uint jm1 = (j == 0u)       ? (NY - 1u) : (j - 1u);
    uint jp1 = (j == NY - 1u)  ? 0u        : (j + 1u);

    uint row_j   = j   * NX;
    uint row_jm1 = jm1 * NX;
    uint row_jp1 = jp1 * NX;

    float f0 = f_in[0u * N + row_j   + i  ];
    float f1 = f_in[1u * N + row_j   + im1];
    float f2 = f_in[2u * N + row_jm1 + i  ];
    float f3 = f_in[3u * N + row_j   + ip1];
    float f4 = f_in[4u * N + row_jp1 + i  ];
    float f5 = f_in[5u * N + row_jm1 + im1];
    float f6 = f_in[6u * N + row_jm1 + ip1];
    float f7 = f_in[7u * N + row_jp1 + ip1];
    float f8 = f_in[8u * N + row_jp1 + im1];

    float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    float inv_rho = 1.0f / rho;
    float mx = f1 - f3 + f5 - f6 - f7 + f8;
    float my = f2 - f4 + f5 + f6 - f7 - f8;
    float ux = mx * inv_rho;
    float uy = my * inv_rho;

    float inv_tau = 1.0f / tau;
    float omt     = 1.0f - inv_tau;

    const float W0 = 4.0f / 9.0f;
    const float W1 = 1.0f / 9.0f;
    const float W5 = 1.0f / 36.0f;

    // Fold rho * inv_tau into prefactors once.
    float rit = rho * inv_tau;
    float A0 = W0 * rit;
    float A1 = W1 * rit;
    float A5 = W5 * rit;

    float usq = fma(ux, ux, uy * uy);
    float c1  = fma(-1.5f, usq, 1.0f);

    // Axis directions: feq = A * (c1 ± 3u + 4.5 u^2)
    float sx = fma(4.5f, ux * ux, c1);
    float tx = 3.0f * ux;
    float feq1 = A1 * (sx + tx);
    float feq3 = A1 * (sx - tx);

    float sy = fma(4.5f, uy * uy, c1);
    float ty = 3.0f * uy;
    float feq2 = A1 * (sy + ty);
    float feq4 = A1 * (sy - ty);

    float feq0 = A0 * c1;

    // Diagonal pair k=5/k=7: cu = ±(ux+uy)
    float d1  = ux + uy;
    float sd1 = fma(4.5f, d1 * d1, c1);
    float td1 = 3.0f * d1;
    float feq5 = A5 * (sd1 + td1);
    float feq7 = A5 * (sd1 - td1);

    // Anti-diagonal pair k=6/k=8: cu = ±(uy-ux)
    float d2  = uy - ux;
    float sd2 = fma(4.5f, d2 * d2, c1);
    float td2 = 3.0f * d2;
    float feq6 = A5 * (sd2 + td2);
    float feq8 = A5 * (sd2 - td2);

    uint idx = row_j + i;

    f_out[0u * N + idx] = fma(omt, f0, feq0);
    f_out[1u * N + idx] = fma(omt, f1, feq1);
    f_out[2u * N + idx] = fma(omt, f2, feq2);
    f_out[3u * N + idx] = fma(omt, f3, feq3);
    f_out[4u * N + idx] = fma(omt, f4, feq4);
    f_out[5u * N + idx] = fma(omt, f5, feq5);
    f_out[6u * N + idx] = fma(omt, f6, feq6);
    f_out[7u * N + idx] = fma(omt, f7, feq7);
    f_out[8u * N + idx] = fma(omt, f8, feq8);
}