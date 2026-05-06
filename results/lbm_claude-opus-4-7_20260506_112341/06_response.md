**Optimization:** Switch threadgroup shape to a 32×2 tile (64 threads, simdgroup-aligned along x). This keeps loads coalesced along the contiguous `i` axis (32 lanes per simdgroup → one 128B transaction per channel per simdgroup), while being narrower in y to keep more rows resident in cache for the y±1 neighbor loads. The small kernels (64×64, 128×128) suffered because 8×8 tiles produced 2 simdgroups partially split across rows, halving load coalescing efficiency. A 32×2 layout guarantees full simdgroup coalescing on x and improves L1/L2 row reuse between the two simdgroups in the threadgroup. I also restructure the equilibrium computation to expose more FMA parallelism and reduce dependency chain length.

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

    // Issue all loads up front for max memory-level parallelism.
    // Lanes within a simdgroup span 32 contiguous i values, so each
    // channel access coalesces into one 128B transaction.
    float f0 = f_in[0u * N + row   + i  ];
    float f1 = f_in[1u * N + row   + im1];
    float f2 = f_in[2u * N + row_m + i  ];
    float f3 = f_in[3u * N + row   + ip1];
    float f4 = f_in[4u * N + row_p + i  ];
    float f5 = f_in[5u * N + row_m + im1];
    float f6 = f_in[6u * N + row_m + ip1];
    float f7 = f_in[7u * N + row_p + ip1];
    float f8 = f_in[8u * N + row_p + im1];

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

    // Equilibrium: t(cu) = A + cu*(3 + 4.5*cu) = A + 3cu + 4.5*cu^2.
    // Using squared form for symmetry: t(±c) shares cu^2.
    float ux2 = ux * ux;
    float uy2 = uy * uy;
    float upv = ux + uy;
    float umv = ux - uy;
    float upv2 = upv * upv;
    float umv2 = umv * umv;

    // A + 4.5*cu^2 (shared between +c and -c).
    float A1 = fma(4.5f, ux2,  A);  // for k=1,3
    float A2 = fma(4.5f, uy2,  A);  // for k=2,4
    float A5 = fma(4.5f, upv2, A);  // for k=5,7
    float A6 = fma(4.5f, umv2, A);  // for k=6,8 (umv = ux-uy = -(uy-ux))

    // t = A1 + 3*cu  for + dir; A1 - 3*cu for - dir.
    float t0 = A;
    float t1 = fma( 3.0f,  ux,  A1);
    float t3 = fma(-3.0f,  ux,  A1);
    float t2 = fma( 3.0f,  uy,  A2);
    float t4 = fma(-3.0f,  uy,  A2);
    float t5 = fma( 3.0f,  upv, A5);
    float t7 = fma(-3.0f,  upv, A5);
    // k=6: cu = uy - ux = -umv;  k=8: cu = ux - uy = umv
    float t8 = fma( 3.0f,  umv, A6);
    float t6 = fma(-3.0f,  umv, A6);

    f_out[0u * N + idx] = fma(one_m_w, f0, orW0 * t0);
    f_out[1u * N + idx] = fma(one_m_w, f1, orWS * t1);
    f_out[2u * N + idx] = fma(one_m_w, f2, orWS * t2);
    f_out[3u * N + idx] = fma(one_m_w, f3, orWS * t3);
    f_out[4u * N + idx] = fma(one_m_w, f4, orWS * t4);
    f_out[5u * N + idx] = fma(one_m_w, f5, orWD * t5);
    f_out[6u * N + idx] = fma(one_m_w, f6, orWD * t6);
    f_out[7u * N + idx] = fma(one_m_w, f7, orWD * t7);
    f_out[8u * N + idx] = fma(one_m_w, f8, orWD * t8);
}
```