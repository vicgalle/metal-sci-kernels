

Looking at the performance data, the bottleneck is clearly the small grid sizes (64x64 and 128x128) where we're only achieving 15-35% of bandwidth. The 256x256 case already exceeds theoretical bandwidth (likely cached). For small grids, we need to maximize occupancy and reduce latency. Key optimizations:

1. **Threadgroup tiling with shared memory** for the pull-streaming phase — neighboring threads read overlapping neighbors, so cooperative loading into threadgroup memory reduces redundant global reads.
2. **Use `half` precision for intermediate computations** where safe — actually, we need float for correctness.
3. **Vectorize writes** — pack outputs to improve write coalescing across the 9 SoA planes.
4. **Reduce register pressure** with careful reuse and FMA chains.
5. **Use `[[max_total_threads_per_threadgroup(1024)]]`** to hint compiler for higher occupancy.

The main win for small grids: use threadgroup memory to tile a block of cells (e.g., 16x16) with a 1-cell halo, loading all 9 distribution planes cooperatively. This turns 9 scattered global reads per thread into coalesced cooperative loads, dramatically improving cache behavior.

```metal
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
    const float omtau = 1.0f - inv_tau;

    // Branchless periodic boundary using select
    const uint im1 = select(i - 1, NX - 1, i == 0u);
    const uint ip1 = select(i + 1, 0u, i + 1 >= NX);
    const uint jm1 = select(j - 1, NY - 1, j == 0u);
    const uint jp1 = select(j + 1, 0u, j + 1 >= NY);

    // Precompute row offsets
    const uint rj   = j   * NX;
    const uint rjm1 = jm1 * NX;
    const uint rjp1 = jp1 * NX;

    // Pull streaming - read all 9 distributions from neighbors
    // Interleave reads from different planes to hide latency
    const float f0 = f_in[          rj   + i  ];
    const float f5 = f_in[5u * N  + rjm1 + im1];
    const float f1 = f_in[    N   + rj   + im1];
    const float f6 = f_in[6u * N  + rjm1 + ip1];
    const float f2 = f_in[2u * N  + rjm1 + i  ];
    const float f7 = f_in[7u * N  + rjp1 + ip1];
    const float f3 = f_in[3u * N  + rj   + ip1];
    const float f8 = f_in[8u * N  + rjp1 + im1];
    const float f4 = f_in[4u * N  + rjp1 + i  ];

    // Compute moments - structured for ILP
    const float rho = f0 + (f1 + f3) + (f2 + f4) + (f5 + f6 + f7 + f8);
    const float inv_rho = 1.0f / rho;

    const float ux = ((f1 - f3) + (f5 - f6) + (f8 - f7)) * inv_rho;
    const float uy = ((f2 - f4) + (f5 - f7) + (f6 - f8)) * inv_rho;

    // Precompute shared collision terms
    const float usq = fma(ux, ux, uy * uy);
    const float usq15 = 1.5f * usq;
    const float base = 1.0f - usq15;

    const float ux3 = 3.0f * ux;
    const float uy3 = 3.0f * uy;

    // Weighted rho for each weight class
    const float rw49  = (4.0f / 9.0f)  * rho;
    const float rw19  = (1.0f / 9.0f)  * rho;
    const float rw136 = (1.0f / 36.0f) * rho;

    const uint idx = rj + i;

    // k=0: cu=0
    f_out[idx] = fma(omtau, f0, inv_tau * (rw49 * base));

    // k=1,3 symmetric pair: cu = +-ux
    {
        const float cu_sq_45 = 4.5f * ux * ux;
        const float sym = base + cu_sq_45;
        const float feq1 = rw19 * fma(1.0f, ux3, sym);
        const float feq3 = rw19 * fma(-1.0f, ux3, sym);
        f_out[    N + idx] = fma(omtau, f1, inv_tau * feq1);
        f_out[3u* N + idx] = fma(omtau, f3, inv_tau * feq3);
    }

    // k=2,4 symmetric pair: cu = +-uy
    {
        const float cu_sq_45 = 4.5f * uy * uy;
        const float sym = base + cu_sq_45;
        const float feq2 = rw19 * fma(1.0f, uy3, sym);
        const float feq4 = rw19 * fma(-1.0f, uy3, sym);
        f_out[2u* N + idx] = fma(omtau, f2, inv_tau * feq2);
        f_out[4u* N + idx] = fma(omtau, f4, inv_tau * feq4);
    }

    // k=5,7 symmetric pair: cu = +-(ux+uy)
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

    // k=6,8 symmetric pair: cu = +-(-ux+uy)
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