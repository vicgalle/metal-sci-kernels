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
    // Pair along x: only even i works; odd i exits.
    if ((i & 1u) != 0u) return;

    uint NXm1 = NX - 1u;
    uint NYm1 = NY - 1u;

    // Branchless periodic wrap.
    uint im1 = (i == 0u)    ? NXm1 : (i - 1u);
    uint i0  = i;
    uint i1  = i + 1u;             // always < NX since NX even-aligned in tests; guard:
    if (i1 >= NX) i1 = 0u;
    uint ip2 = (i1 == NXm1) ? 0u   : (i1 + 1u);
    uint jm1 = (j == 0u)    ? NYm1 : (j - 1u);
    uint jp1 = (j == NYm1)  ? 0u   : (j + 1u);

    uint N = NX * NY;
    uint row_j   = j   * NX;
    uint row_jm1 = jm1 * NX;
    uint row_jp1 = jp1 * NX;

    // Plane bases.
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

    // ---- Cell A: column i0 ----
    {
        float f0 = f_in[b0 + row_j   + i0 ];
        float f1 = f_in[b1 + row_j   + im1];
        float f2 = f_in[b2 + row_jm1 + i0 ];
        float f3 = f_in[b3 + row_j   + i1 ];
        float f4 = f_in[b4 + row_jp1 + i0 ];
        float f5 = f_in[b5 + row_jm1 + im1];
        float f6 = f_in[b6 + row_jm1 + i1 ];
        float f7 = f_in[b7 + row_jp1 + i1 ];
        float f8 = f_in[b8 + row_jp1 + im1];

        float rho = ((f0 + f1) + (f2 + f3)) + ((f4 + f5) + (f6 + f7)) + f8;
        float inv_rho = 1.0f / rho;
        float mx = ((f1 - f3) + (f5 - f6)) + (f8 - f7);
        float my = ((f2 - f4) + (f5 + f6)) - (f7 + f8);
        float ux = mx * inv_rho;
        float uy = my * inv_rho;

        float usq = fma(ux, ux, uy * uy);
        float c1  = fma(-1.5f, usq, 1.0f);
        float rit = rho * inv_tau;
        float A0  = W0 * rit;
        float A1  = W1 * rit;
        float A5  = W5 * rit;

        float sx = fma(4.5f, ux * ux, c1);
        float tx = 3.0f * ux;
        float sy = fma(4.5f, uy * uy, c1);
        float ty = 3.0f * uy;
        float d1 = ux + uy;
        float sd1 = fma(4.5f, d1 * d1, c1);
        float td1 = 3.0f * d1;
        float d2 = uy - ux;
        float sd2 = fma(4.5f, d2 * d2, c1);
        float td2 = 3.0f * d2;

        uint idx = row_j + i0;
        f_out[b0 + idx] = fma(omt, f0, A0 * c1);
        f_out[b1 + idx] = fma(omt, f1, A1 * (sx + tx));
        f_out[b2 + idx] = fma(omt, f2, A1 * (sy + ty));
        f_out[b3 + idx] = fma(omt, f3, A1 * (sx - tx));
        f_out[b4 + idx] = fma(omt, f4, A1 * (sy - ty));
        f_out[b5 + idx] = fma(omt, f5, A5 * (sd1 + td1));
        f_out[b6 + idx] = fma(omt, f6, A5 * (sd2 + td2));
        f_out[b7 + idx] = fma(omt, f7, A5 * (sd1 - td1));
        f_out[b8 + idx] = fma(omt, f8, A5 * (sd2 - td2));
    }

    // If i1 wrapped to 0 (NX==1 edge case) skip second cell to avoid double-write.
    if (i1 == 0u) return;

    // ---- Cell B: column i1 ----
    {
        float f0 = f_in[b0 + row_j   + i1 ];
        float f1 = f_in[b1 + row_j   + i0 ];
        float f2 = f_in[b2 + row_jm1 + i1 ];
        float f3 = f_in[b3 + row_j   + ip2];
        float f4 = f_in[b4 + row_jp1 + i1 ];
        float f5 = f_in[b5 + row_jm1 + i0 ];
        float f6 = f_in[b6 + row_jm1 + ip2];
        float f7 = f_in[b7 + row_jp1 + ip2];
        float f8 = f_in[b8 + row_jp1 + i0 ];

        float rho = ((f0 + f1) + (f2 + f3)) + ((f4 + f5) + (f6 + f7)) + f8;
        float inv_rho = 1.0f / rho;
        float mx = ((f1 - f3) + (f5 - f6)) + (f8 - f7);
        float my = ((f2 - f4) + (f5 + f6)) - (f7 + f8);
        float ux = mx * inv_rho;
        float uy = my * inv_rho;

        float usq = fma(ux, ux, uy * uy);
        float c1  = fma(-1.5f, usq, 1.0f);
        float rit = rho * inv_tau;
        float A0  = W0 * rit;
        float A1  = W1 * rit;
        float A5  = W5 * rit;

        float sx = fma(4.5f, ux * ux, c1);
        float tx = 3.0f * ux;
        float sy = fma(4.5f, uy * uy, c1);
        float ty = 3.0f * uy;
        float d1 = ux + uy;
        float sd1 = fma(4.5f, d1 * d1, c1);
        float td1 = 3.0f * d1;
        float d2 = uy - ux;
        float sd2 = fma(4.5f, d2 * d2, c1);
        float td2 = 3.0f * d2;

        uint idx = row_j + i1;
        f_out[b0 + idx] = fma(omt, f0, A0 * c1);
        f_out[b1 + idx] = fma(omt, f1, A1 * (sx + tx));
        f_out[b2 + idx] = fma(omt, f2, A1 * (sy + ty));
        f_out[b3 + idx] = fma(omt, f3, A1 * (sx - tx));
        f_out[b4 + idx] = fma(omt, f4, A1 * (sy - ty));
        f_out[b5 + idx] = fma(omt, f5, A5 * (sd1 + td1));
        f_out[b6 + idx] = fma(omt, f6, A5 * (sd2 + td2));
        f_out[b7 + idx] = fma(omt, f7, A5 * (sd1 - td1));
        f_out[b8 + idx] = fma(omt, f8, A5 * (sd2 - td2));
    }
}