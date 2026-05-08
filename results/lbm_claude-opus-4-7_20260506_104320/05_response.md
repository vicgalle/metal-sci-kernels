**Optimization:** The small grids (64×64, 128×128) are launch/occupancy-bound, not bandwidth-bound. I'll add `[[max_total_threads_per_threadgroup]]` to encourage better occupancy, and use `select()` for branchless wrap (more SIMD-friendly than ternary on uint compares). I'll also hoist `omega*W*rho` precomputations and use `fma` aggressively to reduce instruction count, freeing the GPU to issue more loads in parallel. Writes are issued contiguously per-k to encourage coalesced burst writes. The 256×256 case already exceeds nominal roofline (cache effects), so the focus is shrinking per-thread instruction latency for the smaller cases.

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

    // Branchless periodic wrap.
    uint ip1 = select(i + 1u, 0u,      i + 1u == nx);
    uint im1 = select(i - 1u, nx - 1u, i == 0u);
    uint jp1 = select(j + 1u, 0u,      j + 1u == ny);
    uint jm1 = select(j - 1u, ny - 1u, j == 0u);

    uint row  = j   * nx;
    uint rowU = jm1 * nx;
    uint rowD = jp1 * nx;

    // Pull-stream loads.
    float f0 = f_in[0u * N + row  + i  ];
    float f1 = f_in[1u * N + row  + im1];
    float f2 = f_in[2u * N + rowU + i  ];
    float f3 = f_in[3u * N + row  + ip1];
    float f4 = f_in[4u * N + rowD + i  ];
    float f5 = f_in[5u * N + rowU + im1];
    float f6 = f_in[6u * N + rowU + ip1];
    float f7 = f_in[7u * N + rowD + ip1];
    float f8 = f_in[8u * N + rowD + im1];

    // Moments.
    float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    float jx  = (f1 - f3) + (f5 - f6) + (f8 - f7);
    float jy  = (f2 - f4) + (f5 + f6) - (f7 + f8);
    float inv_rho = 1.0f / rho;
    float ux = jx * inv_rho;
    float uy = jy * inv_rho;

    float usq   = fma(ux, ux, uy * uy);
    float omega = 1.0f / tau;
    float omm   = 1.0f - omega;
    float c1    = fma(-1.5f, usq, 1.0f);

    const float W0 = 4.0f / 9.0f;
    const float W1 = 1.0f / 9.0f;
    const float W5 = 1.0f / 36.0f;

    float wr0 = omega * W0 * rho;
    float wr1 = omega * W1 * rho;
    float wr5 = omega * W5 * rho;

    uint idx = row + i;

    // k=0
    f_out[0u * N + idx] = fma(omm, f0, wr0 * c1);

    // axis-aligned (W1)
    float cu1 = ux, cu2 = uy, cu3 = -ux, cu4 = -uy;
    float p1 = fma(4.5f * cu1, cu1, fma(3.0f, cu1, c1));
    float p2 = fma(4.5f * cu2, cu2, fma(3.0f, cu2, c1));
    float p3 = fma(4.5f * cu3, cu3, fma(3.0f, cu3, c1));
    float p4 = fma(4.5f * cu4, cu4, fma(3.0f, cu4, c1));
    f_out[1u * N + idx] = fma(omm, f1, wr1 * p1);
    f_out[2u * N + idx] = fma(omm, f2, wr1 * p2);
    f_out[3u * N + idx] = fma(omm, f3, wr1 * p3);
    f_out[4u * N + idx] = fma(omm, f4, wr1 * p4);

    // diagonals (W5)
    float cu5 = ux + uy;
    float cu6 = uy - ux;
    float cu7 = -cu5;
    float cu8 = -cu6;
    float p5 = fma(4.5f * cu5, cu5, fma(3.0f, cu5, c1));
    float p6 = fma(4.5f * cu6, cu6, fma(3.0f, cu6, c1));
    float p7 = fma(4.5f * cu7, cu7, fma(3.0f, cu7, c1));
    float p8 = fma(4.5f * cu8, cu8, fma(3.0f, cu8, c1));
    f_out[5u * N + idx] = fma(omm, f5, wr5 * p5);
    f_out[6u * N + idx] = fma(omm, f6, wr5 * p6);
    f_out[7u * N + idx] = fma(omm, f7, wr5 * p7);
    f_out[8u * N + idx] = fma(omm, f8, wr5 * p8);
}
```