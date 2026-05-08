**Optimization:** The small grids (64x64, 128x128) are dominated by dispatch/launch overhead and underutilization, while the 256x256 case is bandwidth-bound. The incumbent already saturates the math nicely. The main lever left is to (a) reduce instruction count on the small cases by using `precise::divide`/native reciprocal and folding constants more aggressively, and (b) help the compiler issue 9 wide loads/stores cleanly with `int` (signed) arithmetic to avoid unsigned compare overhead in indexing. I also collapse the BGK update to two FMAs per direction using folded prefactors `A_k = W_k * rho * inv_tau` and the combined form `f_out = omt*f + A*(c1 + 3*cu + 4.5*cu^2)` — using `fma` everywhere reduces register pressure and instruction count vs. the incumbent's expanded form.

```metal
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

    uint im1 = (i == 0u)      ? (NX - 1u) : (i - 1u);
    uint ip1 = (i + 1u == NX) ? 0u        : (i + 1u);
    uint jm1 = (j == 0u)      ? (NY - 1u) : (j - 1u);
    uint jp1 = (j + 1u == NY) ? 0u        : (j + 1u);

    uint row_j   = j   * NX;
    uint row_jm1 = jm1 * NX;
    uint row_jp1 = jp1 * NX;

    // Pull-stream loads.
    float f0 = f_in[0u * N + row_j   + i  ];
    float f1 = f_in[1u * N + row_j   + im1];
    float f2 = f_in[2u * N + row_jm1 + i  ];
    float f3 = f_in[3u * N + row_j   + ip1];
    float f4 = f_in[4u * N + row_jp1 + i  ];
    float f5 = f_in[5u * N + row_jm1 + im1];
    float f6 = f_in[6u * N + row_jm1 + ip1];
    float f7 = f_in[7u * N + row_jp1 + ip1];
    float f8 = f_in[8u * N + row_jp1 + im1];

    // Moments.
    float rho = ((f0 + f1) + (f2 + f3)) + ((f4 + f5) + (f6 + f7)) + f8;
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

    // Fold (rho * inv_tau) into per-weight prefactor once.
    float rit = rho * inv_tau;
    float A0  = W0 * rit;
    float A1  = W1 * rit;
    float A5  = W5 * rit;

    // c1 = 1 - 1.5 (ux^2 + uy^2)
    float usq = fma(ux, ux, uy * uy);
    float c1  = fma(-1.5f, usq, 1.0f);

    // For each direction:  f_out = omt*f + A*(c1 + 3 cu + 4.5 cu^2)
    //                            = omt*f + A*( (c1 + 4.5 cu^2) + 3 cu )
    // Rest particle.
    float out0 = fma(omt, f0, A0 * c1);

    // Axis x (k=1: cu=ux, k=3: cu=-ux). 4.5 ux^2 is shared.
    float ux2  = ux * ux;
    float sx   = fma(4.5f, ux2, c1);     // c1 + 4.5 ux^2
    float tx   = 3.0f * ux;
    float out1 = fma(omt, f1, A1 * (sx + tx));
    float out3 = fma(omt, f3, A1 * (sx - tx));

    // Axis y.
    float uy2  = uy * uy;
    float sy   = fma(4.5f, uy2, c1);
    float ty   = 3.0f * uy;
    float out2 = fma(omt, f2, A1 * (sy + ty));
    float out4 = fma(omt, f4, A1 * (sy - ty));

    // Diagonal (k=5: cu=ux+uy, k=7: cu=-(ux+uy)).
    float d1   = ux + uy;
    float sd1  = fma(4.5f, d1 * d1, c1);
    float td1  = 3.0f * d1;
    float out5 = fma(omt, f5, A5 * (sd1 + td1));
    float out7 = fma(omt, f7, A5 * (sd1 - td1));

    // Anti-diagonal (k=6: cu=-ux+uy, k=8: cu=ux-uy).
    float d2   = uy - ux;
    float sd2  = fma(4.5f, d2 * d2, c1);
    float td2  = 3.0f * d2;
    float out6 = fma(omt, f6, A5 * (sd2 + td2));
    float out8 = fma(omt, f8, A5 * (sd2 - td2));

    uint idx = row_j + i;
    f_out[0u * N + idx] = out0;
    f_out[1u * N + idx] = out1;
    f_out[2u * N + idx] = out2;
    f_out[3u * N + idx] = out3;
    f_out[4u * N + idx] = out4;
    f_out[5u * N + idx] = out5;
    f_out[6u * N + idx] = out6;
    f_out[7u * N + idx] = out7;
    f_out[8u * N + idx] = out8;
}
```