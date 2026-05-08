**Optimization:** Algebraic simplification of the BGK collision step. The key insight is that `feq[k]` for opposing directions (e.g., k=1 and k=3) share the same `c1 + 4.5*cu^2` term and only differ in the sign of the `3*cu` term. By computing symmetric/antisymmetric pairs, we halve the multiplications. Additionally, fold `one_minus_inv_tau * f` and `inv_tau * W * rho` constants together. This reduces register pressure and ALU work, letting the kernel get closer to the BW roof on smaller grids where launch overhead and ALU matter. Also use `precise::divide`-free reciprocals via `1.0f/rho` (already fast on Apple GPUs) and move common subexpressions out.

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

    uint im1 = (i == 0u)        ? (NX - 1u) : (i - 1u);
    uint ip1 = (i == NX - 1u)   ? 0u        : (i + 1u);
    uint jm1 = (j == 0u)        ? (NY - 1u) : (j - 1u);
    uint jp1 = (j == NY - 1u)   ? 0u        : (j + 1u);

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
    float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    float inv_rho = 1.0f / rho;
    float mx = (f1 + f5 + f8) - (f3 + f6 + f7);
    float my = (f2 + f5 + f6) - (f4 + f7 + f8);
    float ux = mx * inv_rho;
    float uy = my * inv_rho;

    float usq = ux * ux + uy * uy;
    float inv_tau = 1.0f / tau;
    float omt = 1.0f - inv_tau;

    // Precompute scaled weights * rho * inv_tau so feq contribution = scale * (...).
    const float W0 = 4.0f / 9.0f;
    const float W1 = 1.0f / 9.0f;
    const float W5 = 1.0f / 36.0f;

    float rho_it = rho * inv_tau;
    float A0 = W0 * rho_it;   // for k=0
    float A1 = W1 * rho_it;   // for k=1..4
    float A5 = W5 * rho_it;   // for k=5..8

    float c1 = 1.0f - 1.5f * usq;          // common term

    // k=0: cu=0
    float feq0 = A0 * c1;

    // Axis-aligned pairs share s = c1 + 4.5*u^2; differ in sign of 3*u.
    float sx = c1 + 4.5f * ux * ux;
    float tx = 3.0f * ux;
    float feq1 = A1 * (sx + tx);
    float feq3 = A1 * (sx - tx);

    float sy = c1 + 4.5f * uy * uy;
    float ty = 3.0f * uy;
    float feq2 = A1 * (sy + ty);
    float feq4 = A1 * (sy - ty);

    // Diagonals: cu5 = ux+uy, cu7 = -(ux+uy); share squared term.
    float d1 = ux + uy;
    float sd1 = c1 + 4.5f * d1 * d1;
    float td1 = 3.0f * d1;
    float feq5 = A5 * (sd1 + td1);
    float feq7 = A5 * (sd1 - td1);

    // cu6 = -ux+uy, cu8 = ux-uy; share squared term.
    float d2 = uy - ux;
    float sd2 = c1 + 4.5f * d2 * d2;
    float td2 = 3.0f * d2;
    float feq6 = A5 * (sd2 + td2);
    float feq8 = A5 * (sd2 - td2);

    uint idx = row_j + i;

    f_out[0u * N + idx] = omt * f0 + feq0;
    f_out[1u * N + idx] = omt * f1 + feq1;
    f_out[2u * N + idx] = omt * f2 + feq2;
    f_out[3u * N + idx] = omt * f3 + feq3;
    f_out[4u * N + idx] = omt * f4 + feq4;
    f_out[5u * N + idx] = omt * f5 + feq5;
    f_out[6u * N + idx] = omt * f6 + feq6;
    f_out[7u * N + idx] = omt * f7 + feq7;
    f_out[8u * N + idx] = omt * f8 + feq8;
}
```