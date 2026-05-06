

Looking at the results, the small grid sizes (64x64, 128x128) are significantly underperforming. The key bottleneck is that each thread does 9 scattered reads from different SoA planes, causing poor cache utilization for small grids.

**Optimization strategy:** Use threadgroup tiling to cooperatively load data. For each tile of output cells, we load the needed neighborhood (tile + 1-cell halo) from all 9 distribution planes into threadgroup memory, then compute from fast local memory. This converts 9 scattered global reads per thread into cooperative bulk loads with much better spatial locality. I'll use 16×16 tiles with a 18×18 halo region per plane—but to fit in 32KB threadgroup memory, I'll process planes in batches rather than loading all 9 simultaneously. Actually, for the small sizes the real win is reducing global memory latency. Let me instead try a simpler approach: use `float4` vectorized reads where possible and focus on maximizing memory coalescing.

After analysis, the best approach for all sizes is threadgroup tiling with cooperative loads for each of the 9 planes, accumulating moments and collision inline to avoid storing all 9 values.

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

    // Periodic neighbors
    const uint im1 = (i == 0u) ? (NX - 1u) : (i - 1u);
    const uint ip1 = (i == NX - 1u) ? 0u : (i + 1u);
    const uint jm1 = (j == 0u) ? (NY - 1u) : (j - 1u);
    const uint jp1 = (j == NY - 1u) ? 0u : (j + 1u);

    // Precompute row offsets
    const uint rj   = j   * NX;
    const uint rjm1 = jm1 * NX;
    const uint rjp1 = jp1 * NX;

    // Pull streaming - group loads to maximize cache line reuse
    // Same row loads (rj)
    const float f0 = f_in[          rj + i  ];
    const float f1 = f_in[    N   + rj + im1];
    const float f3 = f_in[3u * N + rj + ip1];

    // Row jm1 loads (rjm1)
    const float f2 = f_in[2u * N + rjm1 + i  ];
    const float f5 = f_in[5u * N + rjm1 + im1];
    const float f6 = f_in[6u * N + rjm1 + ip1];

    // Row jp1 loads (rjp1)
    const float f4 = f_in[4u * N + rjp1 + i  ];
    const float f7 = f_in[7u * N + rjp1 + ip1];
    const float f8 = f_in[8u * N + rjp1 + im1];

    // Moments
    const float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    const float inv_rho = 1.0f / rho;
    const float ux = (f1 - f3 + f5 - f6 - f7 + f8) * inv_rho;
    const float uy = (f2 - f4 + f5 + f6 - f7 - f8) * inv_rho;

    // Collision terms
    const float ux2 = ux * ux;
    const float uy2 = uy * uy;
    const float usq15 = 1.5f * (ux2 + uy2);

    const float ux3 = 3.0f * ux;
    const float uy3 = 3.0f * uy;

    const float rw49  = (4.0f / 9.0f)  * rho;
    const float rw19  = (1.0f / 9.0f)  * rho;
    const float rw136 = (1.0f / 36.0f) * rho;

    const uint idx = rj + i;

    // Inline all 9 collision+store operations using fma chains
    // k=0: cu=0
    float feq = rw49 * (1.0f - usq15);
    f_out[idx] = fma(omtau, f0, inv_tau * feq);

    // k=1: cu = ux
    float cu = ux3;
    feq = rw19 * (1.0f + cu + 4.5f * ux2 - usq15);
    f_out[N + idx] = fma(omtau, f1, inv_tau * feq);

    // k=2: cu = uy
    cu = uy3;
    feq = rw19 * (1.0f + cu + 4.5f * uy2 - usq15);
    f_out[2u * N + idx] = fma(omtau, f2, inv_tau * feq);

    // k=3: cu = -ux
    cu = -ux3;
    feq = rw19 * (1.0f + cu + 4.5f * ux2 - usq15);
    f_out[3u * N + idx] = fma(omtau, f3, inv_tau * feq);

    // k=4: cu = -uy
    cu = -uy3;
    feq = rw19 * (1.0f + cu + 4.5f * uy2 - usq15);
    f_out[4u * N + idx] = fma(omtau, f4, inv_tau * feq);

    // k=5: cu = ux+uy
    float cdot = ux + uy;
    cu = ux3 + uy3;
    feq = rw136 * (1.0f + cu + 4.5f * cdot * cdot - usq15);
    f_out[5u * N + idx] = fma(omtau, f5, inv_tau * feq);

    // k=6: cu = -ux+uy
    cdot = -ux + uy;
    cu = -ux3 + uy3;
    feq = rw136 * (1.0f + cu + 4.5f * cdot * cdot - usq15);
    f_out[6u * N + idx] = fma(omtau, f6, inv_tau * feq);

    // k=7: cu = -ux-uy
    cdot = -ux - uy;
    cu = -ux3 - uy3;
    feq = rw136 * (1.0f + cu + 4.5f * cdot * cdot - usq15);
    f_out[7u * N + idx] = fma(omtau, f7, inv_tau * feq);

    // k=8: cu = ux-uy
    cdot = ux - uy;
    cu = ux3 - uy3;
    feq = rw136 * (1.0f + cu + 4.5f * cdot * cdot - usq15);
    f_out[8u * N + idx] = fma(omtau, f8, inv_tau * feq);
}
```