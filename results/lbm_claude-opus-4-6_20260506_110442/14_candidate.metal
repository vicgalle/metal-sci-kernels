#include <metal_stdlib>
using namespace metal;

#define TILE_W 16
#define TILE_H 16
#define PAD_W (TILE_W + 2)
#define PAD_H (TILE_H + 2)

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
    const float omtau = 1.0f - inv_tau;

    // Periodic neighbors
    const uint im1 = (i > 0u) ? (i - 1u) : (NX - 1u);
    const uint ip1 = (i + 1u < NX) ? (i + 1u) : 0u;
    const uint jm1 = (j > 0u) ? (j - 1u) : (NY - 1u);
    const uint jp1 = (j + 1u < NY) ? (j + 1u) : 0u;

    // Precompute row offsets
    const uint rj   = j   * NX;
    const uint rjm1 = jm1 * NX;
    const uint rjp1 = jp1 * NX;

    // Pull streaming - read all 9 distributions
    // Interleave reads from different planes to help memory-level parallelism
    const float f0 = f_in[          rj   + i  ];
    const float f5 = f_in[5u * N + rjm1 + im1];
    const float f1 = f_in[     N + rj   + im1];
    const float f6 = f_in[6u * N + rjm1 + ip1];
    const float f2 = f_in[2u * N + rjm1 + i  ];
    const float f7 = f_in[7u * N + rjp1 + ip1];
    const float f3 = f_in[3u * N + rj   + ip1];
    const float f8 = f_in[8u * N + rjp1 + im1];
    const float f4 = f_in[4u * N + rjp1 + i  ];

    // Moments - use independent partial sums for ILP
    const float rho = (f0 + f1) + (f2 + f3) + (f4 + f5) + (f6 + f7) + f8;
    const float inv_rho = 1.0f / rho;

    const float ux = ((f1 - f3) + (f5 - f6) + (f8 - f7)) * inv_rho;
    const float uy = ((f2 - f4) + (f5 - f7) + (f6 - f8)) * inv_rho;

    // Precompute collision terms
    const float ux3 = 3.0f * ux;
    const float uy3 = 3.0f * uy;
    const float ux2_45 = 4.5f * ux * ux;
    const float uy2_45 = 4.5f * uy * uy;
    const float usq15 = (ux2_45 + uy2_45) * (1.0f / 3.0f); // = 1.5*(ux2+uy2)
    const float base = 1.0f - usq15;

    // Weight * rho * inv_tau factors
    const float rw49_t  = (4.0f / 9.0f)  * rho * inv_tau;
    const float rw19_t  = (1.0f / 9.0f)  * rho * inv_tau;
    const float rw136_t = (1.0f / 36.0f) * rho * inv_tau;

    const uint idx = rj + i;

    // k=0
    f_out[idx] = fma(omtau, f0, rw49_t * base);

    // k=1,3 (cu = +/- ux)
    {
        const float sym = base + ux2_45;
        f_out[     N + idx] = fma(omtau, f1, rw19_t * (sym + ux3));
        f_out[3u * N + idx] = fma(omtau, f3, rw19_t * (sym - ux3));
    }

    // k=2,4 (cu = +/- uy)
    {
        const float sym = base + uy2_45;
        f_out[2u * N + idx] = fma(omtau, f2, rw19_t * (sym + uy3));
        f_out[4u * N + idx] = fma(omtau, f4, rw19_t * (sym - uy3));
    }

    // k=5,7 (cu = +/-(ux+uy))
    {
        const float cu3 = ux3 + uy3;
        const float cu = ux + uy;
        const float sym = fma(4.5f, cu * cu, base);
        f_out[5u * N + idx] = fma(omtau, f5, rw136_t * (sym + cu3));
        f_out[7u * N + idx] = fma(omtau, f7, rw136_t * (sym - cu3));
    }

    // k=6,8 (cu = +/-(uy-ux))
    {
        const float cu3 = uy3 - ux3;
        const float cu = uy - ux;
        const float sym = fma(4.5f, cu * cu, base);
        f_out[6u * N + idx] = fma(omtau, f6, rw136_t * (sym + cu3));
        f_out[8u * N + idx] = fma(omtau, f8, rw136_t * (sym - cu3));
    }
}