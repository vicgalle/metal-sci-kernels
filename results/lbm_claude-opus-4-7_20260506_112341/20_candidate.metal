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

    // Precompute SoA plane base pointers to keep address math trivial.
    device const float *fp0 = f_in;
    device const float *fp1 = fp0 + N;
    device const float *fp2 = fp1 + N;
    device const float *fp3 = fp2 + N;
    device const float *fp4 = fp3 + N;
    device const float *fp5 = fp4 + N;
    device const float *fp6 = fp5 + N;
    device const float *fp7 = fp6 + N;
    device const float *fp8 = fp7 + N;

    // Pull streaming.
    // CX = {0, 1, 0,-1, 0, 1,-1,-1, 1}
    // CY = {0, 0, 1, 0,-1, 1, 1,-1,-1}
    float f0 = fp0[row   + i  ];
    float f1 = fp1[row   + im1];
    float f2 = fp2[row_m + i  ];
    float f3 = fp3[row   + ip1];
    float f4 = fp4[row_p + i  ];
    float f5 = fp5[row_m + im1];
    float f6 = fp6[row_m + ip1];
    float f7 = fp7[row_p + ip1];
    float f8 = fp8[row_p + im1];

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

    // Output plane pointers.
    device float *gp0 = f_out;
    device float *gp1 = gp0 + N;
    device float *gp2 = gp1 + N;
    device float *gp3 = gp2 + N;
    device float *gp4 = gp3 + N;
    device float *gp5 = gp4 + N;
    device float *gp6 = gp5 + N;
    device float *gp7 = gp6 + N;
    device float *gp8 = gp7 + N;

    // k=0
    gp0[idx] = fma(one_m_w, f0, orW0 * A);

    // k=1: cu = ux
    {
        float cu = ux;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        gp1[idx] = fma(one_m_w, f1, orWS * t);
    }
    // k=2: cu = uy
    {
        float cu = uy;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        gp2[idx] = fma(one_m_w, f2, orWS * t);
    }
    // k=3: cu = -ux
    {
        float cu = -ux;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        gp3[idx] = fma(one_m_w, f3, orWS * t);
    }
    // k=4: cu = -uy
    {
        float cu = -uy;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        gp4[idx] = fma(one_m_w, f4, orWS * t);
    }
    // k=5: cu = ux + uy
    {
        float cu = ux + uy;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        gp5[idx] = fma(one_m_w, f5, orWD * t);
    }
    // k=6: cu = -ux + uy
    {
        float cu = uy - ux;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        gp6[idx] = fma(one_m_w, f6, orWD * t);
    }
    // k=7: cu = -(ux + uy)
    {
        float cu = -(ux + uy);
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        gp7[idx] = fma(one_m_w, f7, orWD * t);
    }
    // k=8: cu = ux - uy
    {
        float cu = ux - uy;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        gp8[idx] = fma(one_m_w, f8, orWD * t);
    }
}