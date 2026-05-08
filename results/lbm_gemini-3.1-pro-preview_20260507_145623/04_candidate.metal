#include <metal_stdlib>
using namespace metal;

kernel void lbm_step(device const float *f_in   [[buffer(0)]],
                     device       float *f_out  [[buffer(1)]],
                     constant uint        &NX   [[buffer(2)]],
                     constant uint        &NY   [[buffer(3)]],
                     constant float       &tau  [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]])
{
    uint x = gid.x;
    uint y = gid.y;
    if (x >= NX || y >= NY) return;

    uint N = NX * NY;

    // Fast periodic boundary wrapping using hardware selects (no modulo)
    uint xm1 = (x > 0) ? x - 1 : NX - 1;
    uint xp1 = (x < NX - 1) ? x + 1 : 0;
    uint ym1 = (y > 0) ? y - 1 : NY - 1;
    uint yp1 = (y < NY - 1) ? y + 1 : 0;

    // Precalculate row offsets
    uint row0  = y * NX;
    uint rowM1 = ym1 * NX;
    uint rowP1 = yp1 * NX;

    // Flatten addresses for 9-way pull streaming
    uint addr[9] = {
        row0  + x,
        row0  + xm1,
        rowM1 + x,
        row0  + xp1,
        rowP1 + x,
        rowM1 + xm1,
        rowM1 + xp1,
        rowP1 + xp1,
        rowP1 + xm1
    };

    // Constant lattice parameters (strictly evaluated at compile-time)
    constexpr float CXf[9] = {0.0f,  1.0f,  0.0f, -1.0f,  0.0f,  1.0f, -1.0f, -1.0f,  1.0f};
    constexpr float CYf[9] = {0.0f,  0.0f,  1.0f,  0.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f};
    constexpr float W[9] = {
        4.0f / 9.0f,
        1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
        1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f
    };

    float f[9];
    float rho = 0.0f;
    float ux  = 0.0f;
    float uy  = 0.0f;

    // 1. Load phase & Macroscopic moments
    // Unrolling allows precise scheduling of independent loads and statically eliminates zero-multiplications.
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        float val = f_in[k * N + addr[k]];
        f[k] = val;
        rho += val;
        ux  += CXf[k] * val;
        uy  += CYf[k] * val;
    }

    float inv_rho = 1.0f / rho;
    ux *= inv_rho;
    uy *= inv_rho;

    float usq = ux * ux + uy * uy;
    float inv_tau = 1.0f / tau;
    uint  idx = row0 + x;
    
    // Shared equilibrium distribution base
    float base = 1.0f - 1.5f * usq;

    // 2. BGK collision & Write phase
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        float cu  = CXf[k] * ux + CYf[k] * uy;
        // Evaluate feq polynomial with optimal FMA mapping structure
        float feq = W[k] * rho * (base + cu * (3.0f + 4.5f * cu));
        f_out[k * N + idx] = f[k] - inv_tau * (f[k] - feq);
    }
}