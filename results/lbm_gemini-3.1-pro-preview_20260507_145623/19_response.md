To push beyond the incumbent, which is heavily memory-bandwidth bound (effectively hitting ~102% of nominal bandwidth due to massive spatial L1 caching), we must ensure that arithmetic instructions and dependent latency do not shadow the memory latency. 

Here we perform a heavy algebraic unrolling of the BGK collision kernel. All runtime arrays (`CX`, `CY`, `W`, `off`) and loop-carried dependencies are fully eliminated, shrinking register pressure. 
We analytically expand the degree-2 BGK polynomials `f_eq`. Opposing velocity vectors only flip signs on their degree-1 terms. This allows us to distill all 9 equilibrium distributions down to just ~15 scalar floating-point operations across shared arithmetic sub-expressions (like `term_p`, `term_x`, etc.). Additionally, tree-based reduction is used for `rho`, `ux`, and `uy` to reuse partial sums and reduce dependent chains. 

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
    uint nx = NX;
    uint ny = NY;
    
    if (i >= nx || j >= ny) return;

    int inx = int(nx);
    int iny = int(ny);
    int N = inx * iny;
    int idx = int(j) * inx + int(i);

    // Fast branchless periodic boundary offset calculation
    int im1 = select(-1, inx - 1, i == 0);
    int ip1 = select(1, 1 - inx, i == nx - 1);
    int jm1 = select(-inx, inx * (iny - 1), j == 0);
    int jp1 = select(inx, -inx * (iny - 1), j == ny - 1);

    device const float *p_in = f_in + idx;
    device       float *p_out = f_out + idx;

    // Fully unrolled pull streaming
    float f0 = p_in[0];
    float f1 = p_in[N + im1];
    float f2 = p_in[2 * N + jm1];
    float f3 = p_in[3 * N + ip1];
    float f4 = p_in[4 * N + jp1];
    float f5 = p_in[5 * N + im1 + jm1];
    float f6 = p_in[6 * N + ip1 + jm1];
    float f7 = p_in[7 * N + ip1 + jp1];
    float f8 = p_in[8 * N + im1 + jp1];

    // Macroscopic moments with instruction-level parallelism
    float f56_sum = f5 + f6;
    float f78_sum = f7 + f8;
    float f56_diff = f5 - f6;
    float f78_diff = f7 - f8;

    float rho = f0 + (f1 + f2) + (f3 + f4) + f56_sum + f78_sum;
    float ux = (f1 - f3) + f56_diff - f78_diff;
    float uy = (f2 - f4) + f56_sum - f78_sum;

    float inv_rho = 1.0f / rho;
    ux *= inv_rho;
    uy *= inv_rho;

    // Algebraic simplification for BGK equilibrium sharing shared factors
    float ux_sq = ux * ux;
    float uy_sq = uy * uy;
    float usq = ux_sq + uy_sq;
    float term0 = fma(-1.5f, usq, 1.0f);

    // Straight directional terms
    float ux3 = 3.0f * ux;
    float uy3 = 3.0f * uy;

    float term_x = fma(4.5f, ux_sq, term0);
    float poly1 = term_x + ux3;
    float poly3 = term_x - ux3;

    float term_y = fma(4.5f, uy_sq, term0);
    float poly2 = term_y + uy3;
    float poly4 = term_y - uy3;

    // Diagonal directional terms
    float term_diag = fma(3.0f, usq, 1.0f);
    float ux_uy_9 = 9.0f * ux * uy;
    
    float term_p = term_diag + ux_uy_9;
    float term_m = term_diag - ux_uy_9;

    float up3 = ux3 + uy3;
    float um3 = ux3 - uy3;

    float poly5 = term_p + up3;
    float poly7 = term_p - up3;
    float poly8 = term_m + um3;
    float poly6 = term_m - um3;

    // BGK Collision execution mapping uniformly to chains
    float inv_tau = 1.0f / tau;
    float oma = 1.0f - inv_tau;
    float rho_tau = rho * inv_tau;

    constexpr float W0 = 4.0f / 9.0f;
    constexpr float W1 = 1.0f / 9.0f;
    constexpr float W5 = 1.0f / 36.0f;

    float w0_rho_tau = W0 * rho_tau;
    float w1_rho_tau = W1 * rho_tau;
    float w5_rho_tau = W5 * rho_tau;

    p_out[0]     = fma(f0, oma, w0_rho_tau * term0);
    p_out[N]     = fma(f1, oma, w1_rho_tau * poly1);
    p_out[2 * N] = fma(f2, oma, w1_rho_tau * poly2);
    p_out[3 * N] = fma(f3, oma, w1_rho_tau * poly3);
    p_out[4 * N] = fma(f4, oma, w1_rho_tau * poly4);
    p_out[5 * N] = fma(f5, oma, w5_rho_tau * poly5);
    p_out[6 * N] = fma(f6, oma, w5_rho_tau * poly6);
    p_out[7 * N] = fma(f7, oma, w5_rho_tau * poly7);
    p_out[8 * N] = fma(f8, oma, w5_rho_tau * poly8);
}
```