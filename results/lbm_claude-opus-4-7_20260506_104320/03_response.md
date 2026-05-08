**Optimization:** Process 2 cells per thread along x-axis with register blocking, and use `select()` for branchless wrap on the inner cell (which has no x-wrap). This halves the dispatch overhead, increases ILP, and lets adjacent reads/writes coalesce better. Each thread loads f_in for two adjacent cells (i, i+1) — many index computations and the `row*nx` arithmetic are amortized. The `i+1` cell needs no left-wrap (since i+1 < nx-1 typically), and the f3 read at i is just f0-style for the next cell shifted — but to keep things simple I just compute both cells inline.

Wait — the grid is dispatched as (NX, NY); I can't change dispatch. Instead, I'll keep the per-thread structure but optimize via SIMD-aligned `fma` and reorganize to reduce register pressure / improve scheduler efficiency. Use `fma` for the equilibrium polynomial and consolidate omega multiplications.

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
    uint nx = NX;
    uint ny = NY;
    if (i >= nx || j >= ny) return;

    uint N  = nx * ny;

    uint ip1 = (i + 1u == nx) ? 0u : (i + 1u);
    uint im1 = (i == 0u) ? (nx - 1u) : (i - 1u);
    uint jp1 = (j + 1u == ny) ? 0u : (j + 1u);
    uint jm1 = (j == 0u) ? (ny - 1u) : (j - 1u);

    uint row    = j   * nx;
    uint rowU   = jm1 * nx;
    uint rowD   = jp1 * nx;

    float f0 = f_in[0u * N + row  + i  ];
    float f1 = f_in[1u * N + row  + im1];
    float f2 = f_in[2u * N + rowU + i  ];
    float f3 = f_in[3u * N + row  + ip1];
    float f4 = f_in[4u * N + rowD + i  ];
    float f5 = f_in[5u * N + rowU + im1];
    float f6 = f_in[6u * N + rowU + ip1];
    float f7 = f_in[7u * N + rowD + ip1];
    float f8 = f_in[8u * N + rowD + im1];

    float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    float jx  = (f1 - f3) + (f5 - f6) + (f8 - f7);
    float jy  = (f2 - f4) + (f5 + f6) - (f7 + f8);
    float inv_rho = 1.0f / rho;
    float ux = jx * inv_rho;
    float uy = jy * inv_rho;

    float usq = ux * ux + uy * uy;
    float omega = 1.0f / tau;
    float one_minus_omega = 1.0f - omega;

    const float W0 = 4.0f / 9.0f;
    const float W1 = 1.0f / 9.0f;
    const float W5 = 1.0f / 36.0f;

    float c1 = 1.0f - 1.5f * usq;

    // Pre-scaled weights: omega * W * rho
    float wr0 = omega * W0 * rho;
    float wr1 = omega * W1 * rho;
    float wr5 = omega * W5 * rho;

    uint idx = row + i;

    // k=0
    f_out[0u * N + idx] = fma(one_minus_omega, f0, wr0 * c1);

    // k=1: cu = ux
    {
        float cu = ux;
        float poly = fma(4.5f * cu, cu, fma(3.0f, cu, c1));
        f_out[1u * N + idx] = fma(one_minus_omega, f1, wr1 * poly);
    }
    // k=2: cu = uy
    {
        float cu = uy;
        float poly = fma(4.5f * cu, cu, fma(3.0f, cu, c1));
        f_out[2u * N + idx] = fma(one_minus_omega, f2, wr1 * poly);
    }
    // k=3: cu = -ux
    {
        float cu = -ux;
        float poly = fma(4.5f * cu, cu, fma(3.0f, cu, c1));
        f_out[3u * N + idx] = fma(one_minus_omega, f3, wr1 * poly);
    }
    // k=4: cu = -uy
    {
        float cu = -uy;
        float poly = fma(4.5f * cu, cu, fma(3.0f, cu, c1));
        f_out[4u * N + idx] = fma(one_minus_omega, f4, wr1 * poly);
    }
    // k=5: cu = ux + uy
    {
        float cu = ux + uy;
        float poly = fma(4.5f * cu, cu, fma(3.0f, cu, c1));
        f_out[5u * N + idx] = fma(one_minus_omega, f5, wr5 * poly);
    }
    // k=6: cu = -ux + uy
    {
        float cu = uy - ux;
        float poly = fma(4.5f * cu, cu, fma(3.0f, cu, c1));
        f_out[6u * N + idx] = fma(one_minus_omega, f6, wr5 * poly);
    }
    // k=7: cu = -ux - uy
    {
        float cu = -(ux + uy);
        float poly = fma(4.5f * cu, cu, fma(3.0f, cu, c1));
        f_out[7u * N + idx] = fma(one_minus_omega, f7, wr5 * poly);
    }
    // k=8: cu = ux - uy
    {
        float cu = ux - uy;
        float poly = fma(4.5f * cu, cu, fma(3.0f, cu, c1));
        f_out[8u * N + idx] = fma(one_minus_omega, f8, wr5 * poly);
    }
}
```