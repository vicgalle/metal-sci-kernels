**Optimization:** Reshape the threadgroup to (32, 2) so each SIMD group works on a contiguous 32-cell row span. This makes the 9 streaming loads coalesce into nicely aligned 128-byte transactions per simdgroup (one per distribution channel), maximizing DRAM bandwidth utilization on the small grids where launch/coalescing overhead dominates. I also keep branchless neighbor selection via `select`, hoist the per-channel base pointers, and fuse the equilibrium term as `cu*(3 + 4.5*cu) + A` with FMAs.

**Why it should help:** The incumbent uses default 2D threadgroup shape, which on Apple GPUs often becomes (NX, 1) capped — for 64×64 this is only 64 threads/TG with poor occupancy and load coalescing. Pinning a 32×2 = 64-thread layout aligns each simdgroup along the contiguous X axis (perfect coalesced loads on `f_in[k*N + row + im1/i/ip1]`) and doubles the rows-per-TG utilization, which should especially lift the 64×64 and 128×128 cases that currently leave bandwidth on the table.

```metal
#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(64)]]
kernel void lbm_step(device const float *f_in   [[buffer(0)]],
                     device       float *f_out  [[buffer(1)]],
                     constant uint        &NX   [[buffer(2)]],
                     constant uint        &NY   [[buffer(3)]],
                     constant float       &tau  [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint j = gid.y;
    if (i >= NX || j >= NY) return;

    uint N  = NX * NY;

    // Branchless periodic neighbors for ±1.
    uint im1 = select(i - 1u, NX - 1u, i == 0u);
    uint ip1 = select(i + 1u, 0u,      i + 1u == NX);
    uint jm1 = select(j - 1u, NY - 1u, j == 0u);
    uint jp1 = select(j + 1u, 0u,      j + 1u == NY);

    uint row    = j   * NX;
    uint row_m  = jm1 * NX;
    uint row_p  = jp1 * NX;

    // Per-channel base pointers (hoisted).
    device const float *b0 = f_in + 0u * N;
    device const float *b1 = f_in + 1u * N;
    device const float *b2 = f_in + 2u * N;
    device const float *b3 = f_in + 3u * N;
    device const float *b4 = f_in + 4u * N;
    device const float *b5 = f_in + 5u * N;
    device const float *b6 = f_in + 6u * N;
    device const float *b7 = f_in + 7u * N;
    device const float *b8 = f_in + 8u * N;

    // Issue all loads up front for max MLP. Within a simdgroup these are
    // contiguous along i (since we pack 32 lanes along x), so each access
    // pattern coalesces into one or two 128B transactions per channel.
    float f0 = b0[row   + i  ];
    float f1 = b1[row   + im1];
    float f2 = b2[row_m + i  ];
    float f3 = b3[row   + ip1];
    float f4 = b4[row_p + i  ];
    float f5 = b5[row_m + im1];
    float f6 = b6[row_m + ip1];
    float f7 = b7[row_p + ip1];
    float f8 = b8[row_p + im1];

    // Moments.
    float rho     = ((f0 + f1) + (f2 + f3)) + ((f4 + f5) + (f6 + f7)) + f8;
    float inv_rho = 1.0f / rho;
    float jx = (f1 - f3) + (f5 + f8) - (f6 + f7);
    float jy = (f2 - f4) + (f5 + f6) - (f7 + f8);
    float ux = jx * inv_rho;
    float uy = jy * inv_rho;

    float usq     = fma(ux, ux, uy * uy);
    float omega   = 1.0f / tau;
    float one_m_w = 1.0f - omega;

    const float W0 = 4.0f / 9.0f;
    const float WS = 1.0f / 9.0f;
    const float WD = 1.0f / 36.0f;

    float orho = omega * rho;
    float orW0 = orho * W0;
    float orWS = orho * WS;
    float orWD = orho * WD;

    // A = 1 - 1.5 * usq
    float A = fma(-1.5f, usq, 1.0f);

    uint idx = row + i;

    // Equilibrium term: t(cu) = A + cu * (3 + 4.5*cu)
    float cu1 = ux;
    float cu2 = uy;
    float cu5 = ux + uy;
    float cu6 = uy - ux;

    float t0 = A;
    float t1 = fma(cu1, fma(4.5f, cu1, 3.0f), A);
    float t2 = fma(cu2, fma(4.5f, cu2, 3.0f), A);
    float t3 = fma(-cu1, fma(4.5f, -cu1, 3.0f), A);
    float t4 = fma(-cu2, fma(4.5f, -cu2, 3.0f), A);
    float t5 = fma(cu5, fma(4.5f, cu5, 3.0f), A);
    float t6 = fma(cu6, fma(4.5f, cu6, 3.0f), A);
    float t7 = fma(-cu5, fma(4.5f, -cu5, 3.0f), A);
    float t8 = fma(-cu6, fma(4.5f, -cu6, 3.0f), A);

    // Per-channel destination pointers.
    device float *o0 = f_out + 0u * N;
    device float *o1 = f_out + 1u * N;
    device float *o2 = f_out + 2u * N;
    device float *o3 = f_out + 3u * N;
    device float *o4 = f_out + 4u * N;
    device float *o5 = f_out + 5u * N;
    device float *o6 = f_out + 6u * N;
    device float *o7 = f_out + 7u * N;
    device float *o8 = f_out + 8u * N;

    o0[idx] = fma(one_m_w, f0, orW0 * t0);
    o1[idx] = fma(one_m_w, f1, orWS * t1);
    o2[idx] = fma(one_m_w, f2, orWS * t2);
    o3[idx] = fma(one_m_w, f3, orWS * t3);
    o4[idx] = fma(one_m_w, f4, orWS * t4);
    o5[idx] = fma(one_m_w, f5, orWD * t5);
    o6[idx] = fma(one_m_w, f6, orWD * t6);
    o7[idx] = fma(one_m_w, f7, orWD * t7);
    o8[idx] = fma(one_m_w, f8, orWD * t8);
}
```