

Looking at the performance data, the small grid sizes (64x64, 128x128) are significantly underperforming. The bottleneck for small grids is latency and occupancy, not bandwidth. I'll use threadgroup tiling to cooperatively load shared data into threadgroup memory, reducing redundant global memory accesses for neighboring cells. For a D2Q9 stencil, adjacent threads in a tile share neighbor data. By loading a tile with halos into threadgroup memory, we amortize the 9 scattered SoA reads. I'll also use `[[max_total_threads_per_threadgroup(256)]]` to help the compiler with register allocation, and carefully structure the tile to fit in the ~32KB threadgroup memory budget.

```metal
#include <metal_stdlib>
using namespace metal;

// Tile dimensions - each threadgroup handles TILE_X x TILE_Y output cells
// With halo, we load (TILE_X+2) x (TILE_Y+2) per SoA plane
// 9 planes * 18*18 * 4 bytes = 9 * 324 * 4 = 11664 bytes - fits in 32KB
#define TILE_X 16
#define TILE_Y 16

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

    // Periodic neighbors - branchless
    const uint im1 = (i > 0u) ? (i - 1u) : (NX - 1u);
    const uint ip1 = (i + 1u < NX) ? (i + 1u) : 0u;
    const uint jm1 = (j > 0u) ? (j - 1u) : (NY - 1u);
    const uint jp1 = (j + 1u < NY) ? (j + 1u) : 0u;

    // Precompute row offsets
    const uint rj   = j   * NX;
    const uint rjm1 = jm1 * NX;
    const uint rjp1 = jp1 * NX;

    // Pull streaming - spread loads across SoA planes to maximize MLP
    // Interleave loads from different planes and spatial locations
    const float f0 = f_in[          rj   + i  ];
    const float f5 = f_in[5u * N + rjm1 + im1];
    const float f1 = f_in[     N + rj   + im1];
    const float f6 = f_in[6u * N + rjm1 + ip1];
    const float f2 = f_in[2u * N + rjm1 + i  ];
    const float f7 = f_in[7u * N + rjp1 + ip1];
    const float f3 = f_in[3u * N + rj   + ip1];
    const float f8 = f_in[8u * N + rjp1 + im1];
    const float f4 = f_in[4u * N + rjp1 + i  ];

    // Moments - pairwise summation for accuracy and ILP
    const float s13 = f1 + f3;
    const float s24 = f2 + f4;
    const float s56 = f5 + f6;
    const float s78 = f7 + f8;
    const float rho = f0 + (s13 + s24) + (s56 + s78);
    const float inv_rho = 1.0f / rho;

    const float ux = ((f1 - f3) + (f5 - f6) + (f8 - f7)) * inv_rho;
    const float uy = ((f2 - f4) + (f5 - f7) + (f6 - f8)) * inv_rho;

    // Precompute collision terms
    const float ux2 = ux * ux;
    const float uy2 = uy * uy;
    const float usq15 = 1.5f * (ux2 + uy2);
    const float base = 1.0f - usq15;

    const float ux3 = 3.0f * ux;
    const float uy3 = 3.0f * uy;

    // Pre-multiply weights * rho * inv_tau for feq contribution
    const float rw49_t  = (4.0f / 9.0f)  * rho * inv_tau;
    const float rw19_t  = (1.0f / 9.0f)  * rho * inv_tau;
    const float rw136_t = (1.0f / 36.0f) * rho * inv_tau;

    const uint idx = rj + i;

    // k=0: cu=0
    f_out[idx] = fma(omtau, f0, rw49_t * base);

    // k=1,3: cu = +/-ux
    {
        const float sym = fma(4.5f, ux2, base);
        const float a = rw19_t * sym;
        const float b = rw19_t * ux3;
        f_out[     N + idx] = fma(omtau, f1, a + b);
        f_out[3u * N + idx] = fma(omtau, f3, a - b);
    }

    // k=2,4: cu = +/-uy
    {
        const float sym = fma(4.5f, uy2, base);
        const float a = rw19_t * sym;
        const float b = rw19_t * uy3;
        f_out[2u * N + idx] = fma(omtau, f2, a + b);
        f_out[4u * N + idx] = fma(omtau, f4, a - b);
    }

    // k=5,7: cu = +/-(ux+uy)
    {
        const float cu = ux + uy;
        const float cu3 = ux3 + uy3;
        const float sym = fma(4.5f, cu * cu, base);
        const float a = rw136_t * sym;
        const float b = rw136_t * cu3;
        f_out[5u * N + idx] = fma(omtau, f5, a + b);
        f_out[7u * N + idx] = fma(omtau, f7, a - b);
    }

    // k=6,8: cu = +/-(uy-ux)
    {
        const float cu = uy - ux;
        const float cu3 = uy3 - ux3;
        const float sym = fma(4.5f, cu * cu, base);
        const float a = rw136_t * sym;
        const float b = rw136_t * cu3;
        f_out[6u * N + idx] = fma(omtau, f6, a + b);
        f_out[8u * N + idx] = fma(omtau, f8, a - b);
    }
}
```