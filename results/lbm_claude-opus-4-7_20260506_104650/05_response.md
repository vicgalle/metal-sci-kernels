**Optimization:** The small grids (64x64, 128x128) are launch/occupancy-bound — they show ~30 GB/s while 256x256 hits ~300 GB/s. To improve small-grid throughput, I'll have each thread process **2 cells along x** (register blocking with width 2). This halves the number of threads launched (better for small grids where launch overhead dominates), reuses the wrap computation for `j±1` rows, and lets adjacent loads share cache lines. I also use `[[max_total_threads_per_threadgroup]]` to hint occupancy and use modular arithmetic via select for cleaner branchless wraps.

The host dispatches a (NX, NY) grid, so I cannot change the grid shape from inside. Instead, I'll keep the original 1-cell-per-thread layout but focus on tightening the inner loop: precompute `inv_tau * W * rho` constants, reuse symmetry between opposite directions (k=1/3, 2/4, 5/7, 6/8) which halves the equilibrium math, and merge the blend into FMAs. This addresses the small-grid bottleneck which is ALU/latency, not BW.

```metal
#include <metal_stdlib>
using namespace metal;

kernel void lbm_step(device const float * __restrict__ f_in   [[buffer(0)]],
                     device       float * __restrict__ f_out  [[buffer(1)]],
                     constant uint        &NX   [[buffer(2)]],
                     constant uint        &NY   [[buffer(3)]],
                     constant float       &tau  [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint j = gid.y;
    if (i >= NX || j >= NY) return;

    uint N = NX * NY;

    uint im1 = select(i - 1u, NX - 1u, i == 0u);
    uint ip1 = select(i + 1u, 0u,      i == NX - 1u);
    uint jm1 = select(j - 1u, NY - 1u, j == 0u);
    uint jp1 = select(j + 1u, 0u,      j == NY - 1u);

    uint row_j   = j   * NX;
    uint row_jm1 = jm1 * NX;
    uint row_jp1 = jp1 * NX;

    // Per-direction base pointers.
    device const float *p0 = f_in;
    device const float *p1 = p0 + N;
    device const float *p2 = p1 + N;
    device const float *p3 = p2 + N;
    device const float *p4 = p3 + N;
    device const float *p5 = p4 + N;
    device const float *p6 = p5 + N;
    device const float *p7 = p6 + N;
    device const float *p8 = p7 + N;

    // Pull-stream loads.
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

    // Pre-multiply weights by rho * inv_tau so feq already includes inv_tau factor.
    float rit = rho * inv_tau;
    float A0  = W0 * rit;
    float A1  = W1 * rit;
    float A5  = W5 * rit;

    float usq = fma(ux, ux, uy * uy);
    float c1  = 1.0f - 1.5f * usq;   // shared (1 - 1.5 u·u) term

    // k=0
    float feq0 = A0 * c1;

    // Axis pairs: feq_pos = A * (c1 + 3 cu + 4.5 cu^2), feq_neg = A * (c1 - 3 cu + 4.5 cu^2)
    // Common: s = c1 + 4.5 cu^2; delta = 3 cu  -> pos = s + delta, neg = s - delta.
    float ux2 = ux * ux;
    float sx  = fma(4.5f, ux2, c1);
    float tx  = 3.0f * ux;
    float feq1 = A1 * (sx + tx);
    float feq3 = A1 * (sx - tx);

    float uy2 = uy * uy;
    float sy  = fma(4.5f, uy2, c1);
    float ty  = 3.0f * uy;
    float feq2 = A1 * (sy + ty);
    float feq4 = A1 * (sy - ty);

    // Diagonal k=5/k=7: cu = ±(ux+uy)
    float d1  = ux + uy;
    float sd1 = fma(4.5f, d1 * d1, c1);
    float td1 = 3.0f * d1;
    float feq5 = A5 * (sd1 + td1);
    float feq7 = A5 * (sd1 - td1);

    // Anti-diagonal k=6/k=8: cu5 = uy-ux for k=6, cu = ux-uy for k=8
    float d2  = uy - ux;
    float sd2 = fma(4.5f, d2 * d2, c1);
    float td2 = 3.0f * d2;
    float feq6 = A5 * (sd2 + td2);
    float feq8 = A5 * (sd2 - td2);

    uint idx = row_j + i;

    // f_out[k] = omt * f[k] + feq[k]  (feq already has inv_tau folded in)
    f_out[0u * N + idx] = fma(omt, f0, feq0);
    f_out[1u * N + idx] = fma(omt, f1, feq1);
    f_out[2u * N + idx] = fma(omt, f2, feq2);
    f_out[3u * N + idx] = fma(omt, f3, feq3);
    f_out[4u * N + idx] = fma(omt, f4, feq4);
    f_out[5u * N + idx] = fma(omt, f5, feq5);
    f_out[6u * N + idx] = fma(omt, f6, feq6);
    f_out[7u * N + idx] = fma(omt, f7, feq7);
    f_out[8u * N + idx] = fma(omt, f8, feq8);
}
```