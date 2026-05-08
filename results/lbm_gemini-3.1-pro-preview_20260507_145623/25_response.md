To resolve the compilation error from the previous attempt, we completely eliminate the need for `CX`, `CY`, and `W` global arrays by fully unrolling the 9 velocity directions into straight-line code. This avoids constant address space issues while dramatically optimizing the kernel.

**Optimizations applied:**
1. **Full Loop Unrolling:** Manually unrolled the streaming, moments, and BGK collision for all 9 directions, replacing dynamic array indexing (`CX[k]`, `CY[k]`) with hardcoded scalar variables (e.g., `cu = ux - uy`).
2. **Algebraic Simplification:** Grouped the equilibrium constants (`term0`, `rho_w`) and fused the BGK collision polynomial to use fewer FMAs.
3. **Safe Pointer Arithmetic:** Switched all indices to 32-bit signed `int` and hoisted constant multipliers for boundary offsets. This avoids 32-to-64-bit zero-extension bugs on negative offsets while minimizing integer ALUs.
4. **Coalesced Memory Bound:** The SoA memory layout naturally provides perfect stride-1 coalescing across the threadgroup, fully saturating the unified memory bandwidth.

```metal
#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(256)]]
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

    int inx = int(nx);
    int iny = int(ny);
    int N = inx * iny;
    int idx = int(j) * inx + int(i);

    // Fast 1D periodic boundary offset calculation, fully hoisted
    int inx_m1 = inx - 1;
    int inx_iny_m1 = inx * (iny - 1);

    int im1 = (i == 0) ? inx_m1 : -1;
    int ip1 = (i == nx - 1) ? -inx_m1 : 1;
    int jm1 = (j == 0) ? inx_iny_m1 : -inx;
    int jp1 = (j == ny - 1) ? -inx_iny_m1 : inx;

    // Stream inputs (fully unrolled, perfectly coalesced memory reads)
    int k_base = idx;
    float v0 = f_in[k_base]; k_base += N;
    float v1 = f_in[k_base + im1]; k_base += N;
    float v2 = f_in[k_base + jm1]; k_base += N;
    float v3 = f_in[k_base + ip1]; k_base += N;
    float v4 = f_in[k_base + jp1]; k_base += N;
    float v5 = f_in[k_base + im1 + jm1]; k_base += N;
    float v6 = f_in[k_base + ip1 + jm1]; k_base += N;
    float v7 = f_in[k_base + ip1 + jp1]; k_base += N;
    float v8 = f_in[k_base + im1 + jp1];

    // Compute moments
    float rho = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8;
    float ux = v1 - v3 + v5 - v6 - v7 + v8;
    float uy = v2 - v4 + v5 + v6 - v7 - v8;

    float inv_rho = 1.0f / rho;
    ux *= inv_rho;
    uy *= inv_rho;

    // BGK Equilibrium common terms
    float usq = ux * ux + uy * uy;
    float inv_tau = 1.0f / tau;
    
    float rho_w0 = rho * (4.0f / 9.0f);
    float rho_w1 = rho * (1.0f / 9.0f);
    float rho_w5 = rho * (1.0f / 36.0f);

    float term0 = 1.0f - 1.5f * usq;

    float t0_w0 = rho_w0 * term0;
    float t0_w1 = rho_w1 * term0;
    float t0_w5 = rho_w5 * term0;

    // Helper macro for collision and output write
    #define CALC_F(v, cu, rw, t0_w) \
        ((v) - inv_tau * ((v) - ((t0_w) + (rw) * ((cu) * 3.0f + (cu) * (cu) * 4.5f))))

    // BGK collision and memory writes (fully unrolled)
    k_base = idx;
    f_out[k_base] = v0 - inv_tau * (v0 - t0_w0); k_base += N;
    f_out[k_base] = CALC_F(v1,  ux,      rho_w1, t0_w1); k_base += N;
    f_out[k_base] = CALC_F(v2,  uy,      rho_w1, t0_w1); k_base += N;
    f_out[k_base] = CALC_F(v3, -ux,      rho_w1, t0_w1); k_base += N;
    f_out[k_base] = CALC_F(v4, -uy,      rho_w1, t0_w1); k_base += N;
    f_out[k_base] = CALC_F(v5,  ux + uy, rho_w5, t0_w5); k_base += N;
    f_out[k_base] = CALC_F(v6, -ux + uy, rho_w5, t0_w5); k_base += N;
    f_out[k_base] = CALC_F(v7, -ux - uy, rho_w5, t0_w5); k_base += N;
    f_out[k_base] = CALC_F(v8,  ux - uy, rho_w5, t0_w5);

    #undef CALC_F
}
```