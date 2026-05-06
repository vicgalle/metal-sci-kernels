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

    // Branchless periodic neighbors.
    uint im1 = (i == 0u)        ? (NX - 1u) : (i - 1u);
    uint ip1 = (i + 1u == NX)   ? 0u        : (i + 1u);
    uint jm1 = (j == 0u)        ? (NY - 1u) : (j - 1u);
    uint jp1 = (j + 1u == NY)   ? 0u        : (j + 1u);

    uint row    = j   * NX;
    uint row_m  = jm1 * NX;
    uint row_p  = jp1 * NX;

    // Issue all 9 loads first to expose maximum memory-level parallelism.
    // CX = {0, 1, 0,-1, 0, 1,-1,-1, 1}
    // CY = {0, 0, 1, 0,-1, 1, 1,-1,-1}
    device const float *p0 = f_in + 0u * N;
    device const float *p1 = f_in + 1u * N;
    device const float *p2 = f_in + 2u * N;
    device const float *p3 = f_in + 3u * N;
    device const float *p4 = f_in + 4u * N;
    device const float *p5 = f_in + 5u * N;
    device const float *p6 = f_in + 6u * N;
    device const float *p7 = f_in + 7u * N;
    device const float *p8 = f_in + 8u * N;

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
    float rho     = ((f0 + f1) + (f2 + f3)) + ((f4 + f5) + (f6 + f7)) + f8;
    float inv_rho = 1.0f / rho;
    float jx = (f1 - f3) + (f5 - f6) + (f8 - f7);
    float jy = (f2 - f4) + (f5 + f6) - (f7 + f8);
    float ux = jx * inv_rho;
    float uy = jy * inv_rho;

    float usq     = fma(ux, ux, uy * uy);
    float omega   = 1.0f / tau;
    float one_m_w = 1.0f - omega;

    const float W0 = 4.0f / 9.0f;
    const float WS = 1.0f / 9.0f;
    const float WD = 1.0f / 36.0f;

    float orho  = omega * rho;
    float orW0  = orho * W0;
    float orWS  = orho * WS;
    float orWD  = orho * WD;

    // A = 1 - 1.5 * usq
    float A = fma(-1.5f, usq, 1.0f);

    uint idx = row + i;

    // For each k: t = A + cu * (3 + 4.5 * cu)
    // f_out = (1 - omega) * f_k + orW * t

    float cu1 = ux;
    float cu2 = uy;
    float cu5 = ux + uy;
    float cu6 = uy - ux;

    float t1 = A + cu1 * fma(4.5f, cu1, 3.0f);
    float t2 = A + cu2 * fma(4.5f, cu2, 3.0f);
    float t3 = A + (-cu1) * fma(4.5f, -cu1, 3.0f);
    float t4 = A + (-cu2) * fma(4.5f, -cu2, 3.0f);
    float t5 = A + cu5 * fma(4.5f, cu5, 3.0f);
    float t6 = A + cu6 * fma(4.5f, cu6, 3.0f);
    float t7 = A + (-cu5) * fma(4.5f, -cu5, 3.0f);
    float t8 = A + (-cu6) * fma(4.5f, -cu6, 3.0f);

    f_out[0u * N + idx] = fma(one_m_w, f0, orW0 * A);
    f_out[1u * N + idx] = fma(one_m_w, f1, orWS * t1);
    f_out[2u * N + idx] = fma(one_m_w, f2, orWS * t2);
    f_out[3u * N + idx] = fma(one_m_w, f3, orWS * t3);
    f_out[4u * N + idx] = fma(one_m_w, f4, orWS * t4);
    f_out[5u * N + idx] = fma(one_m_w, f5, orWD * t5);
    f_out[6u * N + idx] = fma(one_m_w, f6, orWD * t6);
    f_out[7u * N + idx] = fma(one_m_w, f7, orWD * t7);
    f_out[8u * N + idx] = fma(one_m_w, f8, orWD * t8);
}