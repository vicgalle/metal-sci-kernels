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

    uint N  = NX * NY;

    // Branchless periodic neighbors.
    uint im1 = select(i - 1u, NX - 1u, i == 0u);
    uint ip1 = select(i + 1u, 0u,      i + 1u == NX);
    uint jm1 = select(j - 1u, NY - 1u, j == 0u);
    uint jp1 = select(j + 1u, 0u,      j + 1u == NY);

    uint row    = j   * NX;
    uint row_m  = jm1 * NX;
    uint row_p  = jp1 * NX;

    // Pull streaming.  Issue all 9 loads back to back for MLP.
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

    float orW0 = omega * W0 * rho;
    float orWS = omega * WS * rho;
    float orWD = omega * WD * rho;

    float A   = fma(-1.5f, usq, 1.0f);
    float upv = ux + uy;
    float umv = ux - uy;

    float t0 = A;
    float t1 = A + ux  * fma(4.5f, ux,  3.0f);
    float t2 = A + uy  * fma(4.5f, uy,  3.0f);
    float t3 = A - ux  * fma(-4.5f, ux, 3.0f);
    float t4 = A - uy  * fma(-4.5f, uy, 3.0f);
    float t5 = A + upv * fma(4.5f, upv, 3.0f);
    float t6 = A - umv * fma(-4.5f, umv, 3.0f);
    float t7 = A - upv * fma(-4.5f, upv, 3.0f);
    float t8 = A + umv * fma(4.5f, umv, 3.0f);

    uint idx = row + i;
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