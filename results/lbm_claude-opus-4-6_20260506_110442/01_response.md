

Looking at the baseline, the main bottlenecks are:

1. **Scattered global memory reads** due to pull streaming with modulo arithmetic — each of 9 directions requires a separate global load from a different address.
2. **Integer modulo operations** are expensive — can be replaced with branchless conditional subtraction for periodic wrapping.
3. **Loop-based computation** — fully unrolling and using constants directly avoids loop overhead and enables better register allocation.
4. **No threadgroup cooperation** — for the pull step, neighboring threads read overlapping neighborhoods, so tiling into threadgroup memory can reduce redundant global loads.

My optimization strategy: Fully unroll all 9 directions, replace modulo with branchless wrap, precompute constants, and use vectorized float4 stores where possible. For the small grid sizes, threadgroup tiling overhead may not help, so I focus on arithmetic optimization and memory access patterns.

```metal
#include <metal_stdlib>
using namespace metal;

kernel void lbm_step(device const float *f_in   [[buffer(0)]],
                     device       float *f_out  [[buffer(1)]],
                     constant uint        &NX   [[buffer(2)]],
                     constant uint        &NY   [[buffer(3)]],
                     constant float       &tau  [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x;
    const uint j = gid.y;
    if (i >= NX || j >= NY) return;

    const uint N = NX * NY;
    const float inv_tau = 1.0f / tau;

    // Precompute neighbor indices with branchless periodic wrap
    const uint ip1 = (i + 1 < NX) ? (i + 1) : 0;         // i+1 mod NX
    const uint im1 = (i > 0) ? (i - 1) : (NX - 1);       // i-1 mod NX
    const uint jp1 = (j + 1 < NY) ? (j + 1) : 0;         // j+1 mod NY
    const uint jm1 = (j > 0) ? (j - 1) : (NY - 1);       // j-1 mod NY

    // Pull streaming: f_streamed[k] = f_in[k, (i-cx[k]) mod NX, (j-cy[k]) mod NY]
    // Direction k: (cx,cy) -> source is (i-cx, j-cy)
    // k=0: (0,0)  -> (i,   j)
    // k=1: (1,0)  -> (im1, j)
    // k=2: (0,1)  -> (i,   jm1)
    // k=3: (-1,0) -> (ip1, j)
    // k=4: (0,-1) -> (i,   jp1)
    // k=5: (1,1)  -> (im1, jm1)
    // k=6: (-1,1) -> (ip1, jm1)
    // k=7: (-1,-1)-> (ip1, jp1)
    // k=8: (1,-1) -> (im1, jp1)

    const uint row_j   = j   * NX;
    const uint row_jm1 = jm1 * NX;
    const uint row_jp1 = jp1 * NX;

    const float f0 = f_in[          row_j   + i  ];
    const float f1 = f_in[    N   + row_j   + im1];
    const float f2 = f_in[2u* N   + row_jm1 + i  ];
    const float f3 = f_in[3u* N   + row_j   + ip1];
    const float f4 = f_in[4u* N   + row_jp1 + i  ];
    const float f5 = f_in[5u* N   + row_jm1 + im1];
    const float f6 = f_in[6u* N   + row_jm1 + ip1];
    const float f7 = f_in[7u* N   + row_jp1 + ip1];
    const float f8 = f_in[8u* N   + row_jp1 + im1];

    // Moments
    const float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    const float inv_rho = 1.0f / rho;
    // ux = (f1 - f3 + f5 - f6 - f7 + f8) / rho
    // uy = (f2 - f4 + f5 + f6 - f7 - f8) / rho
    const float ux = (f1 - f3 + f5 - f6 - f7 + f8) * inv_rho;
    const float uy = (f2 - f4 + f5 + f6 - f7 - f8) * inv_rho;

    // BGK collision precomputed terms
    const float usq = ux * ux + uy * uy;
    const float usq15 = 1.5f * usq;

    const uint idx = row_j + i;

    // Unrolled collision for each direction
    // f_eq[k] = W[k] * rho * (1 + 3*cu + 4.5*cu^2 - 1.5*usq)
    // f_out[k] = f[k] - inv_tau * (f[k] - f_eq[k])
    // which is: f[k] * (1 - inv_tau) + inv_tau * f_eq[k]
    const float omtau = 1.0f - inv_tau;

    // k=0: cx=0, cy=0, cu=0
    {
        const float feq = (4.0f / 9.0f) * rho * (1.0f - usq15);
        f_out[idx] = fma(omtau, f0, inv_tau * feq);
    }

    // k=1: cx=1, cy=0, cu=ux
    {
        const float cu = ux;
        const float feq = (1.0f / 9.0f) * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - usq15);
        f_out[N + idx] = fma(omtau, f1, inv_tau * feq);
    }

    // k=2: cx=0, cy=1, cu=uy
    {
        const float cu = uy;
        const float feq = (1.0f / 9.0f) * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - usq15);
        f_out[2u*N + idx] = fma(omtau, f2, inv_tau * feq);
    }

    // k=3: cx=-1, cy=0, cu=-ux
    {
        const float cu = -ux;
        const float feq = (1.0f / 9.0f) * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - usq15);
        f_out[3u*N + idx] = fma(omtau, f3, inv_tau * feq);
    }

    // k=4: cx=0, cy=-1, cu=-uy
    {
        const float cu = -uy;
        const float feq = (1.0f / 9.0f) * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - usq15);
        f_out[4u*N + idx] = fma(omtau, f4, inv_tau * feq);
    }

    // k=5: cx=1, cy=1, cu=ux+uy
    {
        const float cu = ux + uy;
        const float feq = (1.0f / 36.0f) * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - usq15);
        f_out[5u*N + idx] = fma(omtau, f5, inv_tau * feq);
    }

    // k=6: cx=-1, cy=1, cu=-ux+uy
    {
        const float cu = -ux + uy;
        const float feq = (1.0f / 36.0f) * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - usq15);
        f_out[6u*N + idx] = fma(omtau, f6, inv_tau * feq);
    }

    // k=7: cx=-1, cy=-1, cu=-ux-uy
    {
        const float cu = -ux - uy;
        const float feq = (1.0f / 36.0f) * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - usq15);
        f_out[7u*N + idx] = fma(omtau, f7, inv_tau * feq);
    }

    // k=8: cx=1, cy=-1, cu=ux-uy
    {
        const float cu = ux - uy;
        const float feq = (1.0f / 36.0f) * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - usq15);
        f_out[8u*N + idx] = fma(omtau, f8, inv_tau * feq);
    }
}
```