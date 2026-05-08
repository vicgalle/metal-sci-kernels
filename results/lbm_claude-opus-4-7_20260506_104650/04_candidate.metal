#include <metal_stdlib>
using namespace metal;

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

    // Branchless periodic wrap for ±1 offsets.
    uint im1 = (i == 0u)       ? (NX - 1u) : (i - 1u);
    uint ip1 = (i == NX - 1u)  ? 0u        : (i + 1u);
    uint jm1 = (j == 0u)       ? (NY - 1u) : (j - 1u);
    uint jp1 = (j == NY - 1u)  ? 0u        : (j + 1u);

    uint row_j   = j   * NX;
    uint row_jm1 = jm1 * NX;
    uint row_jp1 = jp1 * NX;

    // Per-direction base pointers (avoid recomputing k*N).
    device const float *p0 = f_in;
    device const float *p1 = f_in + N;
    device const float *p2 = p1   + N;
    device const float *p3 = p2   + N;
    device const float *p4 = p3   + N;
    device const float *p5 = p4   + N;
    device const float *p6 = p5   + N;
    device const float *p7 = p6   + N;
    device const float *p8 = p7   + N;

    // Pull-stream loads.
    float f0 = p0[row_j   + i  ];
    float f1 = p1[row_j   + im1];
    float f2 = p2[row_jm1 + i  ];
    float f3 = p3[row_j   + ip1];
    float f4 = p4[row_jp1 + i  ];
    float f5 = p5[row_jm1 + im1];
    float f6 = p6[row_jm1 + ip1];
    float f7 = p7[row_jp1 + ip1];
    float f8 = p8[row_jp1 + im1];

    // Moments.
    float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    float inv_rho = 1.0f / rho;
    float mx = (f1 + f8 + f5) - (f3 + f6 + f7);
    float my = (f2 + f5 + f6) - (f4 + f7 + f8);
    float ux = mx * inv_rho;
    float uy = my * inv_rho;

    float usq = ux * ux + uy * uy;
    float inv_tau = 1.0f / tau;
    float omt = 1.0f - inv_tau;

    const float W0 = 4.0f / 9.0f;
    const float W1 = 1.0f / 9.0f;
    const float W5 = 1.0f / 36.0f;

    float rit = rho * inv_tau;       // rho/tau, since feq is multiplied by inv_tau in blend
    float A0 = W0 * rit;
    float A1 = W1 * rit;
    float A5 = W5 * rit;

    float c1 = 1.0f - 1.5f * usq;    // common term

    // k=0
    float feq0 = A0 * c1;

    // x-axis pair (k=1, k=3): cu = ±ux
    float ux2 = ux * ux;
    float sx = c1 + 4.5f * ux2;
    float tx = 3.0f * ux;
    float feq1 = A1 * (sx + tx);
    float feq3 = A1 * (sx - tx);

    // y-axis pair (k=2, k=4): cu = ±uy
    float uy2 = uy * uy;
    float sy = c1 + 4.5f * uy2;
    float ty = 3.0f * uy;
    float feq2 = A1 * (sy + ty);
    float feq4 = A1 * (sy - ty);

    // diagonal (k=5, k=7): cu = ±(ux+uy)
    float d1 = ux + uy;
    float sd1 = c1 + 4.5f * d1 * d1;
    float td1 = 3.0f * d1;
    float feq5 = A5 * (sd1 + td1);
    float feq7 = A5 * (sd1 - td1);

    // anti-diagonal (k=6, k=8): cu = ±(uy-ux)
    float d2 = uy - ux;
    float sd2 = c1 + 4.5f * d2 * d2;
    float td2 = 3.0f * d2;
    float feq6 = A5 * (sd2 + td2);
    float feq8 = A5 * (sd2 - td2);

    uint idx = row_j + i;

    // Output base pointers.
    device float *q0 = f_out;
    device float *q1 = q0 + N;
    device float *q2 = q1 + N;
    device float *q3 = q2 + N;
    device float *q4 = q3 + N;
    device float *q5 = q4 + N;
    device float *q6 = q5 + N;
    device float *q7 = q6 + N;
    device float *q8 = q7 + N;

    // f_out[k] = (1 - 1/tau) * f[k] + (1/tau) * feq[k]
    // With A* = W* * rho/tau, feq* already absorbs the 1/tau factor.
    q0[idx] = fma(omt, f0, feq0);
    q1[idx] = fma(omt, f1, feq1);
    q2[idx] = fma(omt, f2, feq2);
    q3[idx] = fma(omt, f3, feq3);
    q4[idx] = fma(omt, f4, feq4);
    q5[idx] = fma(omt, f5, feq5);
    q6[idx] = fma(omt, f6, feq6);
    q7[idx] = fma(omt, f7, feq7);
    q8[idx] = fma(omt, f8, feq8);
}