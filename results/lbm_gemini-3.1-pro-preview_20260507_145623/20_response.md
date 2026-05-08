To push the performance beyond the incumbent, we need to completely saturate the M-series GPU memory controllers while eliminating any overhead in address generation and arithmetic.

1. **Unsigned Index Arithmetic**: By using purely `uint` for offset calculations and relying on unsigned modular wrapping modulo $2^{32}$ (e.g. `idx + -1u`), we completely eliminate sign-extension instructions (`sxtw`) that the ARM64 compiler otherwise inserts for 64-bit pointer arithmetic.
2. **Maximal ILP & Coalescing**: Manually unrolling all 9 loads cleanly exposes 9 independent memory requests upfront. Because of the SoA layout, adjacent threads naturally coalesce their accesses.
3. **BGK Algebraic Symmetry**: We factor out common subexpressions in the BGK polynomial. Opposite directions share the $4.5(\vec{c}\cdot\vec{u})^2$ term. Additionally, pre-summing terms like `f5+f6` shares work between `rho`, `ux`, and `uy`. The entire collision step shrinks to just ~20 operations. 

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

    uint N = nx * ny;
    uint idx = j * nx + i;

    // Fast 1D purely unsigned periodic boundary offset calculation.
    // Relies on 32-bit unsigned wrapping to natively avoid branching / modulo operators 
    // and zero-cost address generation without ARM64 sign extensions.
    uint im1 = (i == 0)      ? (nx - 1) : -1u;
    uint ip1 = (i == nx - 1) ? (uint)(1 - (int)nx) : 1u;
    uint jm1 = (j == 0)      ? (nx * (ny - 1)) : (uint)(-(int)nx);
    uint jp1 = (j == ny - 1) ? (uint)(-(int)(nx * (ny - 1))) : nx;

    // Pull streamed values from neighbors
    float f0 = f_in[idx];
    float f1 = f_in[N + idx + im1];
    float f2 = f_in[2 * N + idx + jm1];
    float f3 = f_in[3 * N + idx + ip1];
    float f4 = f_in[4 * N + idx + jp1];
    float f5 = f_in[5 * N + idx + im1 + jm1];
    float f6 = f_in[6 * N + idx + ip1 + jm1];
    float f7 = f_in[7 * N + idx + ip1 + jp1];
    float f8 = f_in[8 * N + idx + im1 + jp1];

    // Compute macroscopic moments with optimized instruction grouping
    float f5p6 = f5 + f6;
    float f5m6 = f5 - f6;
    float f7p8 = f7 + f8;
    float f7m8 = f7 - f8;

    float rho = f0 + (f1 + f2) + (f3 + f4) + f5p6 + f7p8;
    float ux  = (f1 - f3) + f5m6 - f7m8;
    float uy  = (f2 - f4) + f5p6 - f7p8;

    float inv_rho = 1.0f / rho;
    ux *= inv_rho;
    uy *= inv_rho;

    // Algebraic simplification for BGK equilibrium
    float ux_sq = ux * ux;
    float uy_sq = uy * uy;
    float usq = ux_sq + uy_sq;
    float term0 = 1.0f - 1.5f * usq;

    // Exploit symmetry: opposite directions share the squared term
    float term_x = term0 + 4.5f * ux_sq;
    float ux3 = 3.0f * ux;
    float eq1 = term_x + ux3;
    float eq3 = term_x - ux3;

    float term_y = term0 + 4.5f * uy_sq;
    float uy3 = 3.0f * uy;
    float eq2 = term_y + uy3;
    float eq4 = term_y - uy3;

    float cu5 = ux + uy;
    float term_xy = term0 + 4.5f * (cu5 * cu5);
    float cu5_3 = 3.0f * cu5;
    float eq5 = term_xy + cu5_3;
    float eq7 = term_xy - cu5_3;

    float cu6 = -ux + uy;
    float term_xmy = term0 + 4.5f * (cu6 * cu6);
    float cu6_3 = 3.0f * cu6;
    float eq6 = term_xmy + cu6_3;
    float eq8 = term_xmy - cu6_3;

    // Collision execution
    float inv_tau_val = 1.0f / tau;
    float oma = 1.0f - inv_tau_val;
    float rt = rho * inv_tau_val;

    float w0_rt = (4.0f / 9.0f) * rt;
    float w1_rt = (1.0f / 9.0f) * rt;
    float w5_rt = (1.0f / 36.0f) * rt;

    f_out[idx]           = f0 * oma + w0_rt * term0;
    f_out[N + idx]       = f1 * oma + w1_rt * eq1;
    f_out[2 * N + idx]   = f2 * oma + w1_rt * eq2;
    f_out[3 * N + idx]   = f3 * oma + w1_rt * eq3;
    f_out[4 * N + idx]   = f4 * oma + w1_rt * eq4;
    f_out[5 * N + idx]   = f5 * oma + w5_rt * eq5;
    f_out[6 * N + idx]   = f6 * oma + w5_rt * eq6;
    f_out[7 * N + idx]   = f7 * oma + w5_rt * eq7;
    f_out[8 * N + idx]   = f8 * oma + w5_rt * eq8;
}
```