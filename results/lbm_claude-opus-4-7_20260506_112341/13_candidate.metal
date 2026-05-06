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
    uint im1 = select(i - 1u, NX - 1u, i == 0u);
    uint ip1 = select(i + 1u, 0u,      i + 1u == NX);
    uint jm1 = select(j - 1u, NY - 1u, j == 0u);
    uint jp1 = select(j + 1u, 0u,      j + 1u == NY);

    uint row    = j   * NX;
    uint row_m  = jm1 * NX;
    uint row_p  = jp1 * NX;

    // Plane base pointers — lets the compiler fold offsets into addressing
    // and (importantly) emit independent address generation per load.
    device const float *p0 = f_in + 0u * N;
    device const float *p1 = f_in + 1u * N;
    device const float *p2 = f_in + 2u * N;
    device const float *p3 = f_in + 3u * N;
    device const float *p4 = f_in + 4u * N;
    device const float *p5 = f_in + 5u * N;
    device const float *p6 = f_in + 6u * N;
    device const float *p7 = f_in + 7u * N;
    device const float *p8 = f_in + 8u * N;

    // Issue all 9 loads early; interleave row planes so consecutive loads
    // hit different cache lines / pages and the LSU can overlap them.
    float f0 = p0[row   + i  ];
    float f2 = p2[row_m + i  ];
    float f4 = p4[row_p + i  ];
    float f1 = p1[row   + im1];
    float f3 = p3[row   + ip1];
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

    float A = fma(-1.5f, usq, 1.0f);

    uint idx = row + i;

    device float *q0 = f_out + 0u * N;
    device float *q1 = f_out + 1u * N;
    device float *q2 = f_out + 2u * N;
    device float *q3 = f_out + 3u * N;
    device float *q4 = f_out + 4u * N;
    device float *q5 = f_out + 5u * N;
    device float *q6 = f_out + 6u * N;
    device float *q7 = f_out + 7u * N;
    device float *q8 = f_out + 8u * N;

    // k=0
    q0[idx] = fma(one_m_w, f0, orW0 * A);

    {
        float cu = ux;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        q1[idx] = fma(one_m_w, f1, orWS * t);
    }
    {
        float cu = uy;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        q2[idx] = fma(one_m_w, f2, orWS * t);
    }
    {
        float cu = -ux;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        q3[idx] = fma(one_m_w, f3, orWS * t);
    }
    {
        float cu = -uy;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        q4[idx] = fma(one_m_w, f4, orWS * t);
    }
    {
        float cu = ux + uy;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        q5[idx] = fma(one_m_w, f5, orWD * t);
    }
    {
        float cu = uy - ux;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        q6[idx] = fma(one_m_w, f6, orWD * t);
    }
    {
        float cu = -(ux + uy);
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        q7[idx] = fma(one_m_w, f7, orWD * t);
    }
    {
        float cu = ux - uy;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        q8[idx] = fma(one_m_w, f8, orWD * t);
    }
}