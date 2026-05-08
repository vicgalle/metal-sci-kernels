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

    uint NXm1 = NX - 1u;
    uint NYm1 = NY - 1u;
    uint N    = NX * NY;

    uint im1 = (i == 0u)    ? NXm1 : (i - 1u);
    uint ip1 = (i == NXm1)  ? 0u   : (i + 1u);
    uint jm1 = (j == 0u)    ? NYm1 : (j - 1u);
    uint jp1 = (j == NYm1)  ? 0u   : (j + 1u);

    uint row_j   = j   * NX;
    uint row_jm1 = jm1 * NX;
    uint row_jp1 = jp1 * NX;

    // Pull-stream loads: 9 distributions from neighbor cells.
    device const float *p0 = f_in;
    device const float *p1 = f_in + N;
    device const float *p2 = p1   + N;
    device const float *p3 = p2   + N;
    device const float *p4 = p3   + N;
    device const float *p5 = p4   + N;
    device const float *p6 = p5   + N;
    device const float *p7 = p6   + N;
    device const float *p8 = p7   + N;

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
    float rho     = ((f0 + f1) + (f2 + f3)) + ((f4 + f5) + (f6 + f7)) + f8;
    float inv_rho = 1.0f / rho;
    float mx = (f1 - f3) + (f5 - f6) + (f8 - f7);
    float my = (f2 - f4) + (f5 + f6) - (f7 + f8);
    float ux = mx * inv_rho;
    float uy = my * inv_rho;

    float inv_tau = 1.0f / tau;
    float omt     = 1.0f - inv_tau;

    const float W0 = 4.0f / 9.0f;
    const float W1 = 1.0f / 9.0f;
    const float W5 = 1.0f / 36.0f;

    float usq = fma(ux, ux, uy * uy);
    float c1  = fma(-1.5f, usq, 1.0f);

    float r0 = rho * (W0 * inv_tau);
    float r1 = rho * (W1 * inv_tau);
    float r5 = rho * (W5 * inv_tau);

    // Pair symmetry: opposite directions share cu^2.
    float ux2 = ux * ux;
    float sx  = fma(4.5f, ux2, c1);
    float tx  = 3.0f * ux;

    float uy2 = uy * uy;
    float sy  = fma(4.5f, uy2, c1);
    float ty  = 3.0f * uy;

    float dpu  = ux + uy;
    float sd1  = fma(4.5f, dpu * dpu, c1);
    float td1  = 3.0f * dpu;

    float dmu  = uy - ux;
    float sd2  = fma(4.5f, dmu * dmu, c1);
    float td2  = 3.0f * dmu;

    uint idx = row_j + i;

    device float *q0 = f_out;
    device float *q1 = f_out + N;
    device float *q2 = q1    + N;
    device float *q3 = q2    + N;
    device float *q4 = q3    + N;
    device float *q5 = q4    + N;
    device float *q6 = q5    + N;
    device float *q7 = q6    + N;
    device float *q8 = q7    + N;

    q0[idx] = fma(omt, f0, r0 * c1);
    q1[idx] = fma(omt, f1, r1 * (sx + tx));
    q2[idx] = fma(omt, f2, r1 * (sy + ty));
    q3[idx] = fma(omt, f3, r1 * (sx - tx));
    q4[idx] = fma(omt, f4, r1 * (sy - ty));
    q5[idx] = fma(omt, f5, r5 * (sd1 + td1));
    q6[idx] = fma(omt, f6, r5 * (sd2 + td2));
    q7[idx] = fma(omt, f7, r5 * (sd1 - td1));
    q8[idx] = fma(omt, f8, r5 * (sd2 - td2));
}