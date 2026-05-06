**Optimization:** Use a 2D threadgroup tile (e.g., 16×8 = 128 threads) with `[[max_total_threads_per_threadgroup(128)]]`. Larger threadgroups improve scheduling efficiency on small grids (64×64) where the prior 64-thread limit caused under-occupancy — that's exactly where the incumbent only hit 33% of peak. Also restructure the periodic-neighbor selects using min/max-style branchless modulo, precompute `omega*rho` weight terms once, and use `fma` more aggressively in the equilibrium polynomial. The store loop is structured so the compiler can issue 9 independent FMAs followed by 9 independent stores. Vectorized index arithmetic stays in registers.

**Why faster:** The incumbent's small-grid performance is occupancy-limited (64×64 = 4096 threads ÷ 64-per-tg = 64 threadgroups, barely filling the GPU). Bumping the threadgroup hint to 128 lets the compiler allocate fewer registers per thread and fit more simdgroups per core, raising small-grid throughput while leaving the BW-bound large-grid case unchanged.

```metal
#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(128)]]
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

    // Branchless periodic neighbors.
    uint im1 = (i == 0u)        ? (NX - 1u) : (i - 1u);
    uint ip1 = (i + 1u == NX)   ? 0u        : (i + 1u);
    uint jm1 = (j == 0u)        ? (NY - 1u) : (j - 1u);
    uint jp1 = (j + 1u == NY)   ? 0u        : (j + 1u);

    uint row    = j   * NX;
    uint row_m  = jm1 * NX;
    uint row_p  = jp1 * NX;

    // Plane base pointers — strength-reduce the k*N multiplies.
    device const float *p0 = f_in;
    device const float *p1 = p0 + N;
    device const float *p2 = p1 + N;
    device const float *p3 = p2 + N;
    device const float *p4 = p3 + N;
    device const float *p5 = p4 + N;
    device const float *p6 = p5 + N;
    device const float *p7 = p6 + N;
    device const float *p8 = p7 + N;

    // Pull streaming. CX={0,1,0,-1,0,1,-1,-1,1}, CY={0,0,1,0,-1,1,1,-1,-1}
    float f0 = p0[row   + i  ];
    float f1 = p1[row   + im1];
    float f2 = p2[row_m + i  ];
    float f3 = p3[row   + ip1];
    float f4 = p4[row_p + i  ];
    float f5 = p5[row_m + im1];
    float f6 = p6[row_m + ip1];
    float f7 = p7[row_p + ip1];
    float f8 = p8[row_p + im1];

    // Moments.
    float rho     = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    float inv_rho = 1.0f / rho;
    float jx = (f1 - f3) + (f5 + f8) - (f6 + f7);
    float jy = (f2 - f4) + (f5 + f6) - (f7 + f8);
    float ux = jx * inv_rho;
    float uy = jy * inv_rho;

    float usq     = ux * ux + uy * uy;
    float omega   = 1.0f / tau;
    float one_m_w = 1.0f - omega;

    const float W0 = 4.0f / 9.0f;
    const float WS = 1.0f / 9.0f;
    const float WD = 1.0f / 36.0f;

    float or_rho = omega * rho;
    float orW0   = or_rho * W0;
    float orWS   = or_rho * WS;
    float orWD   = or_rho * WD;

    float A = fma(-1.5f, usq, 1.0f);

    uint idx = row + i;

    // Compute all 9 outputs first (independent FMAs), then store.
    float o0, o1, o2, o3, o4, o5, o6, o7, o8;

    o0 = fma(one_m_w, f0, orW0 * A);

    {
        float cu = ux;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        o1 = fma(one_m_w, f1, orWS * t);
    }
    {
        float cu = uy;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        o2 = fma(one_m_w, f2, orWS * t);
    }
    {
        float cu = -ux;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        o3 = fma(one_m_w, f3, orWS * t);
    }
    {
        float cu = -uy;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        o4 = fma(one_m_w, f4, orWS * t);
    }
    {
        float cu = ux + uy;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        o5 = fma(one_m_w, f5, orWD * t);
    }
    {
        float cu = uy - ux;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        o6 = fma(one_m_w, f6, orWD * t);
    }
    {
        float cu = -(ux + uy);
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        o7 = fma(one_m_w, f7, orWD * t);
    }
    {
        float cu = ux - uy;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        o8 = fma(one_m_w, f8, orWD * t);
    }

    // Coalesced stores: each thread writes index `idx` in 9 consecutive planes.
    device float *q = f_out + idx;
    q[0]      = o0;
    q[N]      = o1;
    q[2u * N] = o2;
    q[3u * N] = o3;
    q[4u * N] = o4;
    q[5u * N] = o5;
    q[6u * N] = o6;
    q[7u * N] = o7;
    q[8u * N] = o8;
}
```