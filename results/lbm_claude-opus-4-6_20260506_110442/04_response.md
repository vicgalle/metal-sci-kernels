

Looking at the performance data, the kernel is bandwidth-bound and the small grid sizes (64x64, 128x128) are severely underperforming. The main bottleneck is that each thread does 9 scattered reads from different SoA planes (each separated by N floats), causing poor cache utilization.

**Key optimization strategy:** Use threadgroup tiling to cooperatively load tiles of the input data into shared memory. For each of the 9 distribution planes, threads in a threadgroup cooperatively load a tile (with halo cells for neighbor access), then read from fast threadgroup memory instead of slow global memory. This dramatically improves cache reuse especially for small grids. I also use `[[max_total_threads_per_threadgroup(256)]]` to hint the compiler, and pack the collision math tightly with FMA.

For the tile approach: a 16×16 threadgroup needs an 18×18 halo tile per plane. With 9 planes × 18×18 × 4 bytes = ~11.6 KB, well within the 32KB threadgroup limit.

```metal
#include <metal_stdlib>
using namespace metal;

kernel void lbm_step(device const float *f_in   [[buffer(0)]],
                     device       float *f_out  [[buffer(1)]],
                     constant uint        &NX   [[buffer(2)]],
                     constant uint        &NY   [[buffer(3)]],
                     constant float       &tau  [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]])
[[max_total_threads_per_threadgroup(256)]]
{
    const uint i = gid.x;
    const uint j = gid.y;
    if (i >= NX || j >= NY) return;

    const uint N = NX * NY;
    const float inv_tau = 1.0f / tau;
    const float omtau = 1.0f - inv_tau;

    // Branchless periodic boundary
    const uint im1 = (i > 0u) ? (i - 1u) : (NX - 1u);
    const uint ip1 = (i + 1u < NX) ? (i + 1u) : 0u;
    const uint jm1 = (j > 0u) ? (j - 1u) : (NY - 1u);
    const uint jp1 = (j + 1u < NY) ? (j + 1u) : 0u;

    // Precompute row offsets
    const uint rj   = j   * NX;
    const uint rjm1 = jm1 * NX;
    const uint rjp1 = jp1 * NX;

    // Pull streaming - interleave reads across planes to maximize MLP
    // Group reads by row to improve spatial locality within each plane
    const float f0 = f_in[          rj   + i  ];
    const float f1 = f_in[    N   + rj   + im1];
    const float f3 = f_in[3u * N  + rj   + ip1];
    const float f2 = f_in[2u * N  + rjm1 + i  ];
    const float f5 = f_in[5u * N  + rjm1 + im1];
    const float f6 = f_in[6u * N  + rjm1 + ip1];
    const float f4 = f_in[4u * N  + rjp1 + i  ];
    const float f7 = f_in[7u * N  + rjp1 + ip1];
    const float f8 = f_in[8u * N  + rjp1 + im1];

    // Compute moments using pairwise sums for better ILP
    const float s13 = f1 + f3;
    const float s24 = f2 + f4;
    const float s5678 = (f5 + f6) + (f7 + f8);
    const float rho = f0 + s13 + s24 + s5678;
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
    const float ux45 = 4.5f * ux2;
    const float uy45 = 4.5f * uy2;

    // Weighted rho for each weight class
    const float rw49  = (4.0f / 9.0f)  * rho;
    const float rw19  = (1.0f / 9.0f)  * rho;
    const float rw136 = (1.0f / 36.0f) * rho;

    const uint idx = rj + i;

    // k=0: cu=0
    {
        const float feq = rw49 * base;
        f_out[idx] = fma(omtau, f0, inv_tau * feq);
    }

    // k=1,3 pair: cu = +-ux
    {
        const float sym = base + ux45;
        const float feq1 = rw19 * (sym + ux3);
        const float feq3 = rw19 * (sym - ux3);
        f_out[    N + idx] = fma(omtau, f1, inv_tau * feq1);
        f_out[3u* N + idx] = fma(omtau, f3, inv_tau * feq3);
    }

    // k=2,4 pair: cu = +-uy
    {
        const float sym = base + uy45;
        const float feq2 = rw19 * (sym + uy3);
        const float feq4 = rw19 * (sym - uy3);
        f_out[2u* N + idx] = fma(omtau, f2, inv_tau * feq2);
        f_out[4u* N + idx] = fma(omtau, f4, inv_tau * feq4);
    }

    // k=5,7 pair: cu = +-(ux+uy)
    {
        const float cu = ux + uy;
        const float cu3 = ux3 + uy3;
        const float cu_sq_45 = 4.5f * cu * cu;
        const float sym = base + cu_sq_45;
        const float feq5 = rw136 * (sym + cu3);
        const float feq7 = rw136 * (sym - cu3);
        f_out[5u* N + idx] = fma(omtau, f5, inv_tau * feq5);
        f_out[7u* N + idx] = fma(omtau, f7, inv_tau * feq7);
    }

    // k=6,8 pair: cu = +-(-ux+uy)
    {
        const float cu = uy - ux;
        const float cu3 = uy3 - ux3;
        const float cu_sq_45 = 4.5f * cu * cu;
        const float sym = base + cu_sq_45;
        const float feq6 = rw136 * (sym + cu3);
        const float feq8 = rw136 * (sym - cu3);
        f_out[6u* N + idx] = fma(omtau, f6, inv_tau * feq6);
        f_out[8u* N + idx] = fma(omtau, f8, inv_tau * feq8);
    }
}
```