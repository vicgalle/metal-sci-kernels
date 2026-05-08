#include <metal_stdlib>
using namespace metal;

constant float CX_F[9] = {0.0f,  1.0f,  0.0f, -1.0f,  0.0f,  1.0f, -1.0f, -1.0f,  1.0f};
constant float CY_F[9] = {0.0f,  0.0f,  1.0f,  0.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f};
constant float W[9] = {
    4.0f / 9.0f,
    1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
    1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f
};

// Map directional step to ternary index array (0: self, 1: minus, 2: plus)
constant uint CX_IDX[9] = {0, 1, 0, 2, 0, 1, 2, 2, 1};
constant uint CY_IDX[9] = {0, 0, 1, 0, 2, 1, 1, 2, 2};

kernel void lbm_step(device const float *f_in   [[buffer(0)]],
                     device       float *f_out  [[buffer(1)]],
                     constant uint        &NX   [[buffer(2)]],
                     constant uint        &NY   [[buffer(3)]],
                     constant float       &tau  [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]]) 
{
    uint i = gid.x;
    uint j = gid.y;
    if (i >= NX || j >= NY) return;

    uint N = NX * NY;

    // Fast boundary wrapping (eliminates expensive integer modulo division)
    uint x_minus_1 = (i == 0) ? NX - 1 : i - 1;
    uint x_plus_1  = (i + 1 == NX) ? 0 : i + 1;
    uint y_minus_1 = (j == 0) ? NY - 1 : j - 1;
    uint y_plus_1  = (j + 1 == NY) ? 0 : j + 1;

    // Precompute directional offsets into array mappings
    uint xs[3] = { i, x_minus_1, x_plus_1 };
    uint ys[3] = { j * NX, y_minus_1 * NX, y_plus_1 * NX };

    float f[9];
    float rho = 0.0f;
    float ux = 0.0f;
    float uy = 0.0f;

    // Phase 1: Pull streaming & Macroscopic moments
    uint k_N = 0;
    for (int k = 0; k < 9; ++k) {
        // Evaluate neighbor coordinate directly via array index lookup
        uint offset = k_N + ys[CY_IDX[k]] + xs[CX_IDX[k]];
        float val = f_in[offset];
        
        f[k] = val;
        rho += val;
        ux  += CX_F[k] * val;
        uy  += CY_F[k] * val;
        
        k_N += N; // Base loop induction
    }

    float inv_rho = 1.0f / rho;
    ux *= inv_rho;
    uy *= inv_rho;

    // Phase 2: BGK collision evaluation & Store
    float usq_term = 1.0f - 1.5f * (ux * ux + uy * uy);
    float inv_tau = 1.0f / tau;
    uint idx = ys[0] + xs[0]; // Equivalent to j * NX + i

    k_N = 0;
    for (int k = 0; k < 9; ++k) {
        float cu  = CX_F[k] * ux + CY_F[k] * uy;
        // Optimized polynomial FMA chain matching bitwise tolerance criteria
        float feq = W[k] * rho * (usq_term + cu * (3.0f + 4.5f * cu));
        f_out[k_N + idx] = f[k] - inv_tau * (f[k] - feq);
        
        k_N += N;
    }
}