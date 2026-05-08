**Optimization**: The small grids (64x64, 128x128) are dominated by dispatch/launch overhead and tail-effect inefficiency rather than DRAM bandwidth. The 256x256 case already exceeds the nominal 200 GB/s roofline, indicating the bottleneck on small grids is latency, not bandwidth. I'll add `[[max_total_threads_per_threadgroup(64)]]` hint to allow tighter scheduling, use `select()` for branchless wrap, and use `int` arithmetic for index math (which can be cheaper than uint on Apple GPUs in some cases). I'll also precompute the column offset multiplications using a single `mad`. Crucially, I'll write outputs as `float4` packed where alignment allows, but keeping SoA — since SoA forces strided writes, I'll instead just minimize ALU and aggressively use FMA throughout.

```metal
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

    uint N = NX * NY;

    // Branchless periodic wrap using select.
    uint im1 = select(i - 1u, NX - 1u, i == 0u);
    uint ip1 = select(i + 1u, 0u,      i == NX - 1u);
    uint jm1 = select(j - 1u, NY - 1u, j == 0u);
    uint jp1 = select(j + 1u, 0u,      j == NY - 1u);

    uint row_j   = j   * NX;
    uint row_jm1 = jm1 * NX;
    uint row_jp1 = jp1 * NX;

    // Per-direction plane bases.
    uint b0 = 0u;
    uint b1 = N;
    uint b2 = 2u * N;
    uint b3 = 3u * N;
    uint b4 = 4u * N;
    uint b5 = 5u * N;
    uint b6 = 6u * N;
    uint b7 = 7u * N;
    uint b8 = 8u * N;

    float f0 = f_in[b0 + row_j   + i  ];
    float f1 = f_in[b1 + row_j   + im1];
    float f2 = f_in[b2 + row_jm1 + i  ];
    float f3 = f_in[b3 + row_j   + ip1];
    float f4 = f_in[b4 + row_jp1 + i  ];
    float f5 = f_in[b5 + row_jm1 + im1];
    float f6 = f_in[b6 + row_jm1 + ip1];
    float f7 = f_in[b7 + row_jp1 + ip1];
    float f8 = f_in[b8 + row_jp1 + im1];

    // Moments: pair sums for ILP.
    float s01 = f0 + f1;
    float s23 = f2 + f3;
    float s45 = f4 + f5;
    float s67 = f6 + f7;
    float rho = ((s01 + s23) + (s45 + s67)) + f8;
    float inv_rho = 1.0f / rho;

    // mx = f1 - f3 + f5 - f6 - f7 + f8
    float mx = ((f1 - f3) + (f5 - f6)) + (f8 - f7);
    // my = f2 - f4 + f5 + f6 - f7 - f8
    float my = ((f2 - f4) + (f5 + f6)) - (f7 + f8);
    float ux = mx * inv_rho;
    float uy = my * inv_rho;

    float inv_tau = 1.0f / tau;
    float omt     = 1.0f - inv_tau;

    const float W0 = 4.0f / 9.0f;
    const float W1 = 1.0f / 9.0f;
    const float W5 = 1.0f / 36.0f;

    float rit = rho * inv_tau;
    float A0  = W0 * rit;
    float A1  = W1 * rit;
    float A5  = W5 * rit;

    // c1 = 1 - 1.5 (ux^2 + uy^2)
    float usq = fma(ux, ux, uy * uy);
    float c1  = fma(-1.5f, usq, 1.0f);

    // k=0
    float out0 = fma(omt, f0, A0 * c1);

    // Axis x
    float ux2 = ux * ux;
    float sx  = fma(4.5f, ux2, c1);
    float tx  = 3.0f * ux;
    float out1 = fma(omt, f1, A1 * (sx + tx));
    float out3 = fma(omt, f3, A1 * (sx - tx));

    // Axis y
    float uy2 = uy * uy;
    float sy  = fma(4.5f, uy2, c1);
    float ty  = 3.0f * uy;
    float out2 = fma(omt, f2, A1 * (sy + ty));
    float out4 = fma(omt, f4, A1 * (sy - ty));

    // Diagonal k=5/k=7
    float d1  = ux + uy;
    float sd1 = fma(4.5f, d1 * d1, c1);
    float td1 = 3.0f * d1;
    float out5 = fma(omt, f5, A5 * (sd1 + td1));
    float out7 = fma(omt, f7, A5 * (sd1 - td1));

    // Anti-diagonal k=6/k=8
    float d2  = uy - ux;
    float sd2 = fma(4.5f, d2 * d2, c1);
    float td2 = 3.0f * d2;
    float out6 = fma(omt, f6, A5 * (sd2 + td2));
    float out8 = fma(omt, f8, A5 * (sd2 - td2));

    uint idx = row_j + i;
    f_out[b0 + idx] = out0;
    f_out[b1 + idx] = out1;
    f_out[b2 + idx] = out2;
    f_out[b3 + idx] = out3;
    f_out[b4 + idx] = out4;
    f_out[b5 + idx] = out5;
    f_out[b6 + idx] = out6;
    f_out[b7 + idx] = out7;
    f_out[b8 + idx] = out8;
}
```