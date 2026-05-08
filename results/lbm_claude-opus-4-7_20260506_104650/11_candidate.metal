#include <metal_stdlib>
using namespace metal;

#define STRIP 4u

kernel void lbm_step(device const float *f_in   [[buffer(0)]],
                     device       float *f_out  [[buffer(1)]],
                     constant uint        &NX   [[buffer(2)]],
                     constant uint        &NY   [[buffer(3)]],
                     constant float       &tau  [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]]) {
    uint i0 = gid.x * STRIP;
    uint j  = gid.y;
    if (i0 >= NX || j >= NY) return;

    uint NXm1 = NX - 1u;
    uint NYm1 = NY - 1u;
    uint N    = NX * NY;

    uint jm1 = (j == 0u)    ? NYm1 : (j - 1u);
    uint jp1 = (j == NYm1)  ? 0u   : (j + 1u);

    uint row_j   = j   * NX;
    uint row_jm1 = jm1 * NX;
    uint row_jp1 = jp1 * NX;

    uint b0 = 0u;
    uint b1 = N;
    uint b2 = 2u * N;
    uint b3 = 3u * N;
    uint b4 = 4u * N;
    uint b5 = 5u * N;
    uint b6 = 6u * N;
    uint b7 = 7u * N;
    uint b8 = 8u * N;

    float inv_tau = 1.0f / tau;
    float omt     = 1.0f - inv_tau;
    const float W0 = 4.0f / 9.0f;
    const float W1 = 1.0f / 9.0f;
    const float W5 = 1.0f / 36.0f;
    float A0 = W0 * inv_tau;
    float A1 = W1 * inv_tau;
    float A5 = W5 * inv_tau;

    // Process up to STRIP cells along x.
    #pragma clang loop unroll(full)
    for (uint s = 0u; s < STRIP; ++s) {
        uint i = i0 + s;
        if (i >= NX) return;

        uint im1 = (i == 0u)   ? NXm1 : (i - 1u);
        uint ip1 = (i == NXm1) ? 0u   : (i + 1u);

        float f0 = f_in[b0 + row_j   + i  ];
        float f1 = f_in[b1 + row_j   + im1];
        float f2 = f_in[b2 + row_jm1 + i  ];
        float f3 = f_in[b3 + row_j   + ip1];
        float f4 = f_in[b4 + row_jp1 + i  ];
        float f5 = f_in[b5 + row_jm1 + im1];
        float f6 = f_in[b6 + row_jm1 + ip1];
        float f7 = f_in[b7 + row_jp1 + ip1];
        float f8 = f_in[b8 + row_jp1 + im1];

        float rho = ((f0 + f1) + (f2 + f3)) + ((f4 + f5) + (f6 + f7)) + f8;
        float inv_rho = 1.0f / rho;
        float mx = (f1 - f3) + (f5 - f6) + (f8 - f7);
        float my = (f2 - f4) + (f5 + f6) - (f7 + f8);
        float ux = mx * inv_rho;
        float uy = my * inv_rho;

        float usq = fma(ux, ux, uy * uy);
        float c1  = fma(-1.5f, usq, 1.0f);
        float r0  = rho * A0;
        float r1  = rho * A1;
        float r5  = rho * A5;

        float sx  = fma(4.5f, ux * ux, c1);
        float tx  = 3.0f * ux;
        float sy  = fma(4.5f, uy * uy, c1);
        float ty  = 3.0f * uy;
        float d1  = ux + uy;
        float sd1 = fma(4.5f, d1 * d1, c1);
        float td1 = 3.0f * d1;
        float d2  = uy - ux;
        float sd2 = fma(4.5f, d2 * d2, c1);
        float td2 = 3.0f * d2;

        uint idx = row_j + i;
        f_out[b0 + idx] = fma(omt, f0, r0 * c1);
        f_out[b1 + idx] = fma(omt, f1, r1 * (sx + tx));
        f_out[b2 + idx] = fma(omt, f2, r1 * (sy + ty));
        f_out[b3 + idx] = fma(omt, f3, r1 * (sx - tx));
        f_out[b4 + idx] = fma(omt, f4, r1 * (sy - ty));
        f_out[b5 + idx] = fma(omt, f5, r5 * (sd1 + td1));
        f_out[b6 + idx] = fma(omt, f6, r5 * (sd2 + td2));
        f_out[b7 + idx] = fma(omt, f7, r5 * (sd1 - td1));
        f_out[b8 + idx] = fma(omt, f8, r5 * (sd2 - td2));
    }
}