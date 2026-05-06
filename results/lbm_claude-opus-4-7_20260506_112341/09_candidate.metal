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

    // Precompute base pointers per slice to reduce index arithmetic.
    device const float *p0 = f_in + 0u * N;
    device const float *p1 = f_in + 1u * N;
    device const float *p2 = f_in + 2u * N;
    device const float *p3 = f_in + 3u * N;
    device const float *p4 = f_in + 4u * N;
    device const float *p5 = f_in + 5u * N;
    device const float *p6 = f_in + 6u * N;
    device const float *p7 = f_in + 7u * N;
    device const float *p8 = f_in + 8u * N;

    // Pull streaming — issue all 9 loads back-to-back for MLP.
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

    float orW0 = omega * W0 * rho;
    float orWS = omega * WS * rho;
    float orWD = omega * WD * rho;

    // A = 1 - 1.5 * usq
    float A = fma(-1.5f, usq, 1.0f);

    uint idx = row + i;

    // k=0
    f_out[0u * N + idx] = fma(one_m_w, f0, orW0 * A);

    // Cardinal directions
    {
        float t1 = A + ux * fma(4.5f,  ux, 3.0f);
        f_out[1u * N + idx] = fma(one_m_w, f1, orWS * t1);
    }
    {
        float t2 = A + uy * fma(4.5f,  uy, 3.0f);
        f_out[2u * N + idx] = fma(one_m_w, f2, orWS * t2);
    }
    {
        float nux = -ux;
        float t3  = A + nux * fma(4.5f, nux, 3.0f);
        f_out[3u * N + idx] = fma(one_m_w, f3, orWS * t3);
    }
    {
        float nuy = -uy;
        float t4  = A + nuy * fma(4.5f, nuy, 3.0f);
        f_out[4u * N + idx] = fma(one_m_w, f4, orWS * t4);
    }

    // Diagonals
    float upv = ux + uy;
    float umv = ux - uy;
    {
        float t5 = A + upv * fma(4.5f, upv, 3.0f);
        f_out[5u * N + idx] = fma(one_m_w, f5, orWD * t5);
    }
    {
        float nu = -umv;  // -ux + uy
        float t6 = A + nu * fma(4.5f, nu, 3.0f);
        f_out[6u * N + idx] = fma(one_m_w, f6, orWD * t6);
    }
    {
        float nu = -upv;
        float t7 = A + nu * fma(4.5f, nu, 3.0f);
        f_out[7u * N + idx] = fma(one_m_w, f7, orWD * t7);
    }
    {
        float t8 = A + umv * fma(4.5f, umv, 3.0f);
        f_out[8u * N + idx] = fma(one_m_w, f8, orWD * t8);
    }
}