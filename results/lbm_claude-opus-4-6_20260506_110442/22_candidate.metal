#include <metal_stdlib>
using namespace metal;

kernel void lbm_step(device const float *f_in   [[buffer(0)]],
                     device       float *f_out  [[buffer(1)]],
                     constant uint        &NX   [[buffer(2)]],
                     constant uint        &NY   [[buffer(3)]],
                     constant float       &tau  [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]])
{
    const uint i = gid.x;
    const uint j = gid.y;
    if (i >= NX || j >= NY) return;

    const uint N = NX * NY;
    const float inv_tau = 1.0f / tau;

    // Periodic neighbors
    const uint im1 = i > 0u ? i - 1u : NX - 1u;
    const uint ip1 = i + 1u < NX ? i + 1u : 0u;
    const uint jm1 = j > 0u ? j - 1u : NY - 1u;
    const uint jp1 = j + 1u < NY ? j + 1u : 0u;

    // Row offsets
    const uint rj   = j   * NX;
    const uint rjm1 = jm1 * NX;
    const uint rjp1 = jp1 * NX;

    // Pull streaming - interleave loads from different SoA planes
    // to maximize memory-level parallelism across cache lines
    const uint N2 = N + N;
    const uint N3 = N2 + N;
    const uint N4 = N3 + N;
    const uint N5 = N4 + N;
    const uint N6 = N5 + N;
    const uint N7 = N6 + N;
    const uint N8 = N7 + N;

    const float f0 = f_in[       rj   + i  ];
    const float f5 = f_in[N5 +  rjm1 + im1];
    const float f1 = f_in[N  +  rj   + im1];
    const float f6 = f_in[N6 +  rjm1 + ip1];
    const float f2 = f_in[N2 +  rjm1 + i  ];
    const float f7 = f_in[N7 +  rjp1 + ip1];
    const float f3 = f_in[N3 +  rj   + ip1];
    const float f8 = f_in[N8 +  rjp1 + im1];
    const float f4 = f_in[N4 +  rjp1 + i  ];

    // Moments with pairwise summation for ILP
    const float p13 = f1 + f3;
    const float p24 = f2 + f4;
    const float p56 = f5 + f6;
    const float p78 = f7 + f8;
    const float rho = f0 + (p13 + p24) + (p56 + p78);
    const float inv_rho = 1.0f / rho;

    const float ux = ((f1 - f3) + (f5 - f6) + (f8 - f7)) * inv_rho;
    const float uy = ((f2 - f4) + (f5 - f7) + (f6 - f8)) * inv_rho;

    // Precompute shared collision terms
    const float ux2 = ux * ux;
    const float uy2 = uy * uy;
    const float usq15 = 1.5f * (ux2 + uy2);
    const float base = 1.0f - usq15;
    const float ux3 = 3.0f * ux;
    const float uy3 = 3.0f * uy;

    // Combine (1 - 1/tau) and (w*rho/tau) factors
    const float omtau = 1.0f - inv_tau;
    const float rt49  = (4.0f / 9.0f)  * rho * inv_tau;
    const float rt19  = (1.0f / 9.0f)  * rho * inv_tau;
    const float rt136 = (1.0f / 36.0f) * rho * inv_tau;

    const uint idx = rj + i;

    // k=0
    f_out[idx] = fma(omtau, f0, rt49 * base);

    // k=1,3 (cu = +/-ux)
    {
        const float sym = fma(4.5f, ux2, base);
        const float a = rt19 * sym;
        const float b = rt19 * ux3;
        f_out[N  + idx] = fma(omtau, f1, a + b);
        f_out[N3 + idx] = fma(omtau, f3, a - b);
    }

    // k=2,4 (cu = +/-uy)
    {
        const float sym = fma(4.5f, uy2, base);
        const float a = rt19 * sym;
        const float b = rt19 * uy3;
        f_out[N2 + idx] = fma(omtau, f2, a + b);
        f_out[N4 + idx] = fma(omtau, f4, a - b);
    }

    // k=5,7 (cu = +/-(ux+uy))
    {
        const float cu = ux + uy;
        const float sym = fma(4.5f, cu * cu, base);
        const float a = rt136 * sym;
        const float b = rt136 * (ux3 + uy3);
        f_out[N5 + idx] = fma(omtau, f5, a + b);
        f_out[N7 + idx] = fma(omtau, f7, a - b);
    }

    // k=6,8 (cu = +/-(uy-ux))
    {
        const float cu = uy - ux;
        const float sym = fma(4.5f, cu * cu, base);
        const float a = rt136 * sym;
        const float b = rt136 * (uy3 - ux3);
        f_out[N6 + idx] = fma(omtau, f6, a + b);
        f_out[N8 + idx] = fma(omtau, f8, a - b);
    }
}