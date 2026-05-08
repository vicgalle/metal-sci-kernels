#include <metal_stdlib>
using namespace metal;

constexpr constant float CX[9] = {0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 1.0f, -1.0f, -1.0f, 1.0f};
constexpr constant float CY[9] = {0.0f, 0.0f, 1.0f, 0.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f};
constexpr constant float W[9] = {
    4.0f / 9.0f,
    1.0f / 9.0f,  1.0f / 9.0f,  1.0f / 9.0f,  1.0f / 9.0f,
    1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f
};

kernel void lbm_step(device const float *f_in   [[buffer(0)]],
                     device       float *f_out  [[buffer(1)]],
                     constant uint        &NX   [[buffer(2)]],
                     constant uint        &NY   [[buffer(3)]],
                     constant float       &tau  [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint j = gid.y;
    uint nx = NX;
    uint ny = NY;
    
    if (i >= nx || j >= ny) return;

    uint N = nx * ny;

    // Fast periodic boundary wrapping using select instead of modulo division
    uint im1 = (i > 0) ? i - 1 : nx - 1;
    uint ip1 = (i == nx - 1) ? 0 : i + 1;
    uint jm1 = (j > 0) ? j - 1 : ny - 1;
    uint jp1 = (j == ny - 1) ? 0 : j + 1;

    uint j_nx   = j * nx;
    uint jm1_nx = jm1 * nx;
    uint jp1_nx = jp1 * nx;

    // Precompute perfectly aligned 1D offsets for pull streaming
    uint idx_src[9];
    idx_src[0] = j_nx   + i;
    idx_src[1] = j_nx   + im1;
    idx_src[2] = jm1_nx + i;
    idx_src[3] = j_nx   + ip1;
    idx_src[4] = jp1_nx + i;
    idx_src[5] = jm1_nx + im1;
    idx_src[6] = jm1_nx + ip1;
    idx_src[7] = jp1_nx + ip1;
    idx_src[8] = jp1_nx + im1;

    float f[9];
    float rho = 0.0f;
    float ux  = 0.0f;
    float uy  = 0.0f;

    // Load streamed values and accumulate macroscopic moments
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        float val = f_in[k * N + idx_src[k]];
        f[k] = val;
        rho += val;
        ux  += CX[k] * val;
        uy  += CY[k] * val;
    }

    float inv_rho = 1.0f / rho;
    ux *= inv_rho;
    uy *= inv_rho;

    // BGK collision optimized with Horner's method for the polynomial
    float usq = ux * ux + uy * uy;
    float usq_term = 1.0f - 1.5f * usq;
    float inv_tau = 1.0f / tau;
    uint idx = j_nx + i;

    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        float cu = CX[k] * ux + CY[k] * uy;
        float feq = W[k] * rho * (usq_term + cu * (3.0f + 4.5f * cu));
        f_out[k * N + idx] = f[k] - inv_tau * (f[k] - feq);
    }
}