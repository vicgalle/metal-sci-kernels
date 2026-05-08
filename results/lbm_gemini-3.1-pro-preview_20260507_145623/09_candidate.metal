#include <metal_stdlib>
using namespace metal;

constant float CX_f[9] = {0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 1.0f, -1.0f, -1.0f, 1.0f};
constant float CY_f[9] = {0.0f, 0.0f, 1.0f, 0.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f};
constant float W[9] = {
    4.0f / 9.0f,
    1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
    1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f
};
constant uint dir_x[9] = {0, 1, 0, 2, 0, 1, 2, 2, 1};
constant uint dir_y[9] = {0, 0, 1, 0, 2, 1, 1, 2, 2};

kernel void lbm_step(device const float * restrict f_in   [[buffer(0)]],
                     device       float * restrict f_out  [[buffer(1)]],
                     constant uint        &NX   [[buffer(2)]],
                     constant uint        &NY   [[buffer(3)]],
                     constant float       &tau  [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]]) 
{
    uint i = gid.x;
    uint j = gid.y;
    uint nx = NX;
    uint ny = NY;
    
    if (i >= nx || j >= ny) return;

    uint N = nx * ny;

    // Fast periodic boundary wrapping using select instead of modulo
    uint x_m1 = (i == 0) ? nx - 1 : i - 1;
    uint x_p1 = (i + 1 == nx) ? 0 : i + 1;
    uint y_m1 = (j == 0) ? ny - 1 : j - 1;
    uint y_p1 = (j + 1 == ny) ? 0 : j + 1;

    // Precompute spatial source coordinates for all directions
    // Index mapping: 0 -> center, 1 -> minus 1, 2 -> plus 1
    uint src_x[3]  = { i, x_m1, x_p1 };
    uint base_y[3] = { j * nx, y_m1 * nx, y_p1 * nx };

    float f[9];
    float rho = 0.0f;
    float ux = 0.0f;
    float uy = 0.0f;

    // 1. Pull streaming + Moments
    // With unrolling, constant offsets fold into immediate literals and 
    // zero-multiplications vanish at compile-time.
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        float val = f_in[k * N + base_y[dir_y[k]] + src_x[dir_x[k]]];
        f[k] = val;
        rho += val;
        ux  += CX_f[k] * val;
        uy  += CY_f[k] * val;
    }

    float inv_rho = 1.0f / rho;
    ux *= inv_rho;
    uy *= inv_rho;

    // 2. BGK collision
    float usq = ux * ux + uy * uy;
    float usq_term = 1.0f - 1.5f * usq;
    
    float tau_val = tau;
    float inv_tau = 1.0f / tau_val;
    float tau_comp = 1.0f - inv_tau;
    
    uint idx = base_y[0] + i;

    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        float cu = CX_f[k] * ux + CY_f[k] * uy;
        // Factored equivalent to: 1 + 3cu + 4.5cu^2 - 1.5usq
        float feq = W[k] * rho * (usq_term + cu * (3.0f + 4.5f * cu));
        f_out[k * N + idx] = f[k] * tau_comp + feq * inv_tau;
    }
}