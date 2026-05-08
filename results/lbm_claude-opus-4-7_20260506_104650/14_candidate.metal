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

    uint im1 = select(i - 1u, NXm1, i == 0u);
    uint ip1 = select(i + 1u, 0u,   i == NXm1);
    uint jm1 = select(j - 1u, NYm1, j == 0u);
    uint jp1 = select(j + 1u, 0u,   j == NYm1);

    uint row_j   = j   * NX;
    uint row_jm1 = jm1 * NX;
    uint row_jp1 = jp1 * NX;

    uint b1 = N;
    uint b2 = b1 + N;
    uint b3 = b2 + N;
    uint b4 = b3 + N;
    uint b5 = b4 + N;
    uint b6 = b5 + N;
    uint b7 = b6 + N;
    uint b8 = b7 + N;

    float f0 = f_in[          row_j   + i  ];
    float f1 = f_in[b1 +      row_j   + im1];
    float f2 = f_in[b2 +      row_jm1 + i  ];
    float f3 = f_in[b3 +      row_j   + ip1];
    float f4 = f_in[b4 +      row_jp1 + i  ];
    float f5 = f_in[b5 +      row_jm1 + im1];
    float f6 = f_in[b6 +      row_jm1 + ip1];
    float f7 = f_in[b7 +      row_jp1 + ip1];
    float f8 = f_in[b8 +      row_jp1 + im1];

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

    // f_out[k] = omt*f[k] + inv_tau*W[k]*rho*(c1 + 3*cu + 4.5*cu^2)
    // For opposite pairs, cu flips sign so cu^2 is shared.
    float usq = fma(ux, ux, uy * uy);
    float c1  = fma(-1.5f, usq, 1.0f);

    float r0 = rho * (W0 * inv_tau);
    float r1 = rho * (W1 * inv_tau);
    float r5 = rho * (W5 * inv_tau);

    // k=1,3 pair: cu = ±ux
    float ux2 = ux * ux;
    float sx  = fma(4.5f, ux2, c1);  // c1 + 4.5*ux^2
    float tx  = 3.0f * ux;

    // k=2,4 pair: cu = ±uy
    float uy2 = uy * uy;
    float sy  = fma(4.5f, uy2, c1);
    float ty  = 3.0f * uy;

    // k=5,7 pair: cu = ±(ux+uy)
    float dpu  = ux + uy;
    float dpu2 = dpu * dpu;
    float sd1  = fma(4.5f, dpu2, c1);
    float td1  = 3.0f * dpu;

    // k=6,8 pair: cu = ±(uy-ux)  (k6 = -ux+uy, k8 = ux-uy = -(uy-ux))
    float dmu  = uy - ux;
    float dmu2 = dmu * dmu;
    float sd2  = fma(4.5f, dmu2, c1);
    float td2  = 3.0f * dmu;

    uint idx = row_j + i;

    f_out[      idx] = fma(omt, f0, r0 * c1);
    f_out[b1 +  idx] = fma(omt, f1, r1 * (sx + tx));
    f_out[b2 +  idx] = fma(omt, f2, r1 * (sy + ty));
    f_out[b3 +  idx] = fma(omt, f3, r1 * (sx - tx));
    f_out[b4 +  idx] = fma(omt, f4, r1 * (sy - ty));
    f_out[b5 +  idx] = fma(omt, f5, r5 * (sd1 + td1));
    f_out[b6 +  idx] = fma(omt, f6, r5 * (sd2 + td2));
    f_out[b7 +  idx] = fma(omt, f7, r5 * (sd1 - td1));
    f_out[b8 +  idx] = fma(omt, f8, r5 * (sd2 - td2));
}