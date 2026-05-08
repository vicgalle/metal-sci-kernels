To optimize this memory-bound LBM kernel, we eliminate the computationally expensive integer modulo operations `(src + N) % N` used in the incumbent for periodic boundary conditions. Instead, we use branchless ternary select instructions (`i == 0 ? NX - 1 : i - 1`) which compile to single-cycle `select` ALU operations. This eliminates the integer division bottleneck that throttled smaller grid sizes.

Additionally, we leverage the compiler's loop unrolling and constant folding by wrapping the BGK collision step in a `#pragma unroll` loop and defining the velocity vectors `CX`, `CY` as `constexpr`. LLVM fully unrolls this and trivially folds the zero-multiplications (e.g., `0.0f * ux`), dramatically reducing the total instruction count while strictly preserving mathematically identical behavior and avoiding the register spilling issues of a manual expansion. 

```metal
#include <metal_stdlib>
using namespace metal;

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
    
    // Fast, branchless periodic boundary wrap
    uint im1 = (i == 0) ? NX - 1 : i - 1;
    uint ip1 = (i == NX - 1) ? 0 : i + 1;
    uint jm1 = (j == 0) ? NY - 1 : j - 1;
    uint jp1 = (j == NY - 1) ? 0 : j + 1;

    // Precalculate row offsets
    uint jNX   = j * NX;
    uint jm1NX = jm1 * NX;
    uint jp1NX = jp1 * NX;

    // 1. Pull streaming with perfectly coalesced SoA loads
    float f[9];
    f[0] = f_in[          jNX   + i];
    f[1] = f_in[1 * N + jNX   + im1];
    f[2] = f_in[2 * N + jm1NX + i];
    f[3] = f_in[3 * N + jNX   + ip1];
    f[4] = f_in[4 * N + jp1NX + i];
    f[5] = f_in[5 * N + jm1NX + im1];
    f[6] = f_in[6 * N + jm1NX + ip1];
    f[7] = f_in[7 * N + jp1NX + ip1];
    f[8] = f_in[8 * N + jp1NX + im1];

    // 2. Compute macroscopic moments
    float rho = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7] + f[8];
    float ux  = f[1] - f[3] + f[5] - f[6] - f[7] + f[8];
    float uy  = f[2] - f[4] + f[5] + f[6] - f[7] - f[8];
    
    float inv_rho = 1.0f / rho;
    ux *= inv_rho;
    uy *= inv_rho;

    // 3. BGK collision
    float usq = ux * ux + uy * uy;
    float inv_tau = 1.0f / tau;
    uint  idx = jNX + i;

    constexpr float CX[9] = {0.0f,  1.0f,  0.0f, -1.0f,  0.0f,  1.0f, -1.0f, -1.0f,  1.0f};
    constexpr float CY[9] = {0.0f,  0.0f,  1.0f,  0.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f};
    constexpr float W[9] = {
        4.0f / 9.0f,
        1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
        1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f
    };

    // Explicitly unroll loop; the compiler automatically constant-folds 
    // the zero-multiplications (e.g. 0.0f * ux) and reuses common subexpressions.
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        float cu  = CX[k] * ux + CY[k] * uy;
        float feq = W[k] * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * usq);
        f_out[k * N + idx] = f[k] - inv_tau * (f[k] - feq);
    }
}
```