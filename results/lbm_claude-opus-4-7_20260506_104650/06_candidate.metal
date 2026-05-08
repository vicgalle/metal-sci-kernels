#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(256)]]
kernel void lbm_step(device const float * __restrict__ f_in   [[buffer(0)]],
                     device       float * __restrict__ f_out  [[buffer(1)]],
                     constant uint        &NX   [[buffer(2)]],
                     constant uint        &NY   [[buffer(3)]],
                     constant float       &tau  [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint j = gid.y;
    if (i >= NX || j >= NY) return;

    uint N = NX * NY;

    uint im1 = (i == 0u)      ? (NX - 1u) : (i - 1u);
    uint ip1 = (i == NX - 1u) ? 0u        : (i + 1u);
    uint jm1 = (j == 0u)      ? (NY - 1u) : (j - 1u);
    uint jp1 = (j == NY - 1u) ? 0u        : (j + 1u);

    uint row_j   = j   * NX;
    uint row_jm1 = jm1 * NX;
    uint row_jp1 = jp1 * NX;

    // Per-direction base pointers — lets compiler fold k*N into immediate offsets.
    device const float *b0 = f_in;
    device const float *b1 = b0 + N;
    device const float *b2 = b1 + N;
    device const float *b3 = b2 + N;
    device const float *b4 = b3 + N;
    device const float *b5 = b4 + N;
    device const float *b6 = b5 + N;
    device const float *b7 = b6 + N;
    device const float *b8 = b7 + N;

    // Pull-stream loads.
    float f0 = b0[row_j   + i  ];
    float f1 = b1[row_j   + im1];
    float f2 = b2[row_jm1 + i  ];
    float f3 = b3[row_j   + ip1];
    float f4 = b4[row_jp1 + i  ];
    float f5 = b5[row_jm1 + im1];
    float f6 = b6[row_jm1 + ip1];
    float f7 = b7[row_jp1 + ip1];
    float f8 = b8[row_jp1 + im1];

    // Moments.
    float rho = (f0 + f1 + f2 + f3) + (f4 + f5 + f6 + f7) + f8;
    float inv_rho = 1.0f / rho;
    float mx = (f1 - f3) + (f5 - f6) + (f8 - f7);
    float my = (f2 - f4) + (f5 + f6) - (f7 + f8);
    float ux = mx * inv_rho;
    float uy = my * inv_rho;

    float inv_tau = 1.0f / tau;
    float omt     = 1.0f - inv_tau;

    // Fold rho * inv_tau into the equilibrium prefactors.
    float rit = rho * inv_tau;
    const float W0 = 4.0f / 9.0f;
    const float W1 = 1.0f / 9.0f;
    const float W5 = 1.0f / 36.0f;
    float A0 = W0 * rit;
    float A1 = W1 * rit;
    float A5 = W5 * rit;

    float usq = fma(ux, ux, uy * uy);
    float c1  = fma(-1.5f, usq, 1.0f);     // 1 - 1.5 u·u

    // k=0
    float feq0 = A0 * c1;

    // Axis pairs share s = c1 + 4.5*cu^2; pos = s + 3cu, neg = s - 3cu.
    float sx  = fma(4.5f, ux * ux, c1);
    float tx  = 3.0f * ux;
    float feq1 = A1 * (sx + tx);
    float feq3 = A1 * (sx - tx);

    float sy  = fma(4.5f, uy * uy, c1);
    float ty  = 3.0f * uy;
    float feq2 = A1 * (sy + ty);
    float feq4 = A1 * (sy - ty);

    // Diagonals k=5 (cu=ux+uy) / k=7 (cu=-(ux+uy)).
    float d1  = ux + uy;
    float sd1 = fma(4.5f, d1 * d1, c1);
    float td1 = 3.0f * d1;
    float feq5 = A5 * (sd1 + td1);
    float feq7 = A5 * (sd1 - td1);

    // Anti-diagonals k=6 (cu=uy-ux) / k=8 (cu=ux-uy).
    float d2  = uy - ux;
    float sd2 = fma(4.5f, d2 * d2, c1);
    float td2 = 3.0f * d2;
    float feq6 = A5 * (sd2 + td2);
    float feq8 = A5 * (sd2 - td2);

    uint idx = row_j + i;

    // Per-direction output base pointers.
    device float *o0 = f_out;
    device float *o1 = o0 + N;
    device float *o2 = o1 + N;
    device float *o3 = o2 + N;
    device float *o4 = o3 + N;
    device float *o5 = o4 + N;
    device float *o6 = o5 + N;
    device float *o7 = o6 + N;
    device float *o8 = o7 + N;

    // f_out = omt * f + feq*  (where feq* already includes inv_tau).
    o0[idx] = fma(omt, f0, feq0);
    o1[idx] = fma(omt, f1, feq1);
    o2[idx] = fma(omt, f2, feq2);
    o3[idx] = fma(omt, f3, feq3);
    o4[idx] = fma(omt, f4, feq4);
    o5[idx] = fma(omt, f5, feq5);
    o6[idx] = fma(omt, f6, feq6);
    o7[idx] = fma(omt, f7, feq7);
    o8[idx] = fma(omt, f8, feq8);
}