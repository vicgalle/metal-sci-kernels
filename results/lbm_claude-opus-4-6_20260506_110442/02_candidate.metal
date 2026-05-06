#include <metal_stdlib>
using namespace metal;

kernel void lbm_step(device const float *f_in   [[buffer(0)]],
                     device       float *f_out  [[buffer(1)]],
                     constant uint        &NX   [[buffer(2)]],
                     constant uint        &NY   [[buffer(3)]],
                     constant float       &tau  [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]])
// [[max_total_threads_per_threadgroup(256)]]
{
    const uint i = gid.x;
    const uint j = gid.y;
    if (i >= NX || j >= NY) return;

    const uint N = NX * NY;
    const float inv_tau = 1.0f / tau;
    const float omtau = 1.0f - inv_tau;

    // Branchless periodic wrap using modular arithmetic
    // For pull streaming: source = (i - cx, j - cy) mod (NX, NY)
    // Use select() for branchless conditionals
    const uint ip1 = select(i + 1, 0u, i + 1 >= NX);
    const uint im1 = select(i - 1, NX - 1, i == 0u);
    const uint jp1 = select(j + 1, 0u, j + 1 >= NY);
    const uint jm1 = select(j - 1, NY - 1, j == 0u);

    const uint row_j   = j   * NX;
    const uint row_jm1 = jm1 * NX;
    const uint row_jp1 = jp1 * NX;

    // Pull streaming - gather from neighbors
    const float f0 = f_in[          row_j   + i  ];
    const float f1 = f_in[    N   + row_j   + im1];
    const float f2 = f_in[2u* N   + row_jm1 + i  ];
    const float f3 = f_in[3u* N   + row_j   + ip1];
    const float f4 = f_in[4u* N   + row_jp1 + i  ];
    const float f5 = f_in[5u* N   + row_jm1 + im1];
    const float f6 = f_in[6u* N   + row_jm1 + ip1];
    const float f7 = f_in[7u* N   + row_jp1 + ip1];
    const float f8 = f_in[8u* N   + row_jp1 + im1];

    // Compute moments using partial sums to exploit ILP
    const float f13 = f1 + f3;
    const float f24 = f2 + f4;
    const float f5678 = f5 + f6 + f7 + f8;
    const float rho = f0 + f13 + f24 + f5678;
    const float inv_rho = 1.0f / rho;

    // Velocity: group terms for better ILP
    const float ux = (f1 - f3 + f5 - f6 - f7 + f8) * inv_rho;
    const float uy = (f2 - f4 + f5 + f6 - f7 - f8) * inv_rho;

    // Precompute shared terms
    const float usq = fma(ux, ux, uy * uy);
    const float usq15 = 1.5f * usq;
    const float base = 1.0f - usq15;

    // Precompute weighted rho values
    const float rho_w19  = (1.0f / 9.0f) * rho;
    const float rho_w49  = (4.0f / 9.0f) * rho;
    const float rho_w136 = (1.0f / 36.0f) * rho;

    // Precompute 3*ux, 3*uy for reuse
    const float ux3 = 3.0f * ux;
    const float uy3 = 3.0f * uy;

    const uint idx = row_j + i;

    // k=0: cu=0
    {
        const float feq = rho_w49 * base;
        f_out[idx] = fma(omtau, f0, inv_tau * feq);
    }

    // k=1: cu=ux;  k=3: cu=-ux  (symmetric pair)
    {
        const float cu_sq_45 = 4.5f * ux * ux;
        const float sym = base + cu_sq_45;
        const float feq1 = rho_w19 * (sym + ux3);
        const float feq3 = rho_w19 * (sym - ux3);
        f_out[N + idx]    = fma(omtau, f1, inv_tau * feq1);
        f_out[3u*N + idx] = fma(omtau, f3, inv_tau * feq3);
    }

    // k=2: cu=uy;  k=4: cu=-uy  (symmetric pair)
    {
        const float cu_sq_45 = 4.5f * uy * uy;
        const float sym = base + cu_sq_45;
        const float feq2 = rho_w19 * (sym + uy3);
        const float feq4 = rho_w19 * (sym - uy3);
        f_out[2u*N + idx] = fma(omtau, f2, inv_tau * feq2);
        f_out[4u*N + idx] = fma(omtau, f4, inv_tau * feq4);
    }

    // k=5: cu=ux+uy;  k=7: cu=-ux-uy  (symmetric pair)
    {
        const float cu = ux + uy;
        const float cu3 = ux3 + uy3;
        const float cu_sq_45 = 4.5f * cu * cu;
        const float sym = base + cu_sq_45;
        const float feq5 = rho_w136 * (sym + cu3);
        const float feq7 = rho_w136 * (sym - cu3);
        f_out[5u*N + idx] = fma(omtau, f5, inv_tau * feq5);
        f_out[7u*N + idx] = fma(omtau, f7, inv_tau * feq7);
    }

    // k=6: cu=-ux+uy;  k=8: cu=ux-uy  (symmetric pair)
    {
        const float cu = -ux + uy;
        const float cu3 = -ux3 + uy3;
        const float cu_sq_45 = 4.5f * cu * cu;
        const float sym = base + cu_sq_45;
        const float feq6 = rho_w136 * (sym + cu3);
        const float feq8 = rho_w136 * (sym - cu3);
        f_out[6u*N + idx] = fma(omtau, f6, inv_tau * feq6);
        f_out[8u*N + idx] = fma(omtau, f8, inv_tau * feq8);
    }
}