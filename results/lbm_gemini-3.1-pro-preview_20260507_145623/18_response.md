This optimization reduces both branching overhead and ALU operation count while maximizing memory coalescence given the rigid SoA layout constraints. 

1. **Eliminating Branches**: The periodic boundary conditions are vectorized using Metal's `select()` function. This compiles directly to fast conditional selectors (`csel` equivalents) which avoid branch divergence penalties on the GPU execution pipelines.
2. **ALU Optimization**: The BGK equation is algebraically simplified to extract multiplications. By rewriting `f_out = f - (1/tau) * (f - feq)` into an explicit chained Float-Multiply-Add (FMA) series: `fma(f, 1 - 1/tau, feq * 1/tau)`, we map collision directly to hardware FMAs.
3. **Weight Pre-computation**: Multiplying the collision scalar values out allows 6 scalar multiplications to be hoisted/eliminated per thread. 

```metal
#include <metal_stdlib>
using namespace metal;

constexpr float CX[9] = {0.0f,  1.0f,  0.0f, -1.0f,  0.0f,  1.0f, -1.0f, -1.0f,  1.0f};
constexpr float CY[9] = {0.0f,  0.0f,  1.0f,  0.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f};

kernel void lbm_step(device const float *f_in   [[buffer(0)]],
                     device       float *f_out  [[buffer(1)]],
                     constant uint        &NX   [[buffer(2)]],
                     constant uint        &NY   [[buffer(3)]],
                     constant float       &tau  [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]])
                     [[max_total_threads_per_threadgroup(256)]]
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

    const int off[9] = {
        0, im1, jm1, ip1, jp1, im1 + jm1, ip1 + jm1, ip1 + jp1, im1 + jp1
    };

    device const float *p_in = f_in + idx;
    device       float *p_out = f_out + idx;

    float f[9];
    float rho = 0.0f;
    float ux = 0.0f;
    float uy = 0.0f;

    // Pull streaming and compute raw moments
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        float v = p_in[k * N + off[k]];
        f[k] = v;
        rho += v;
        ux += CX[k] * v;
        uy += CY[k] * v;
    }

    // Normalize moments
    float inv_rho = 1.0f / rho;
    ux *= inv_rho;
    uy *= inv_rho;

    // Derived scalars for algebraic simplification of BGK
    float usq = fma(ux, ux, uy * uy);
    float inv_tau = 1.0f / tau;
    float oma = 1.0f - inv_tau;
    float rho_tau = rho * inv_tau;
    float term0 = fma(-1.5f, usq, 1.0f);

    constexpr float W0 = 4.0f / 9.0f;
    constexpr float W1 = 1.0f / 9.0f;
    constexpr float W5 = 1.0f / 36.0f;

    float w0_rho_tau = W0 * rho_tau;
    float w1_rho_tau = W1 * rho_tau;
    float w5_rho_tau = W5 * rho_tau;

    const float w_rho_tau[9] = {
        w0_rho_tau,
        w1_rho_tau, w1_rho_tau, w1_rho_tau, w1_rho_tau,
        w5_rho_tau, w5_rho_tau, w5_rho_tau, w5_rho_tau
    };

    // Collision execution explicitly resolving to chained FMAs
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        float cu = CX[k] * ux + CY[k] * uy;
        float poly = fma(cu, fma(cu, 4.5f, 3.0f), term0);
        float feq_tau = w_rho_tau[k] * poly;
        p_out[k * N] = fma(f[k], oma, feq_tau);
    }
}
```