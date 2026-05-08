To optimize this bandwidth-bound LBM kernel, we need to maximize memory access efficiency while keeping the ALU instruction count and register footprint minimal so that maximum occupancy can be maintained. 

1. **`constexpr` Arrays**: By declaring the lattice velocity components `CX, CY` and weights `W` as `constexpr`, LLVM handles them as compile-time immediates. When the loops are unrolled, terms multiply by 0 or 1 vanish or simplify instantly, completely dissolving the need to load them from memory or registers.
2. **Algebraic Factorization**: The BGK collision term is analytically refactored to `f[k] * (1 - 1/tau) + (f_eq / tau)`. This avoids redundant subtractions. The `cu` polynomial `1 + 3cu + 4.5cu^2 - 1.5u^2` is refactored into `(1 - 1.5u^2) + cu * (3 + 4.5cu)`, saving one FMA instruction per lattice direction.
3. **Array SROA**: The boundary offset array `off` combined with `#pragma unroll` triggers LLVM's Scalar Replacement of Aggregates (SROA), effectively inlining the boundary logic without explicit branching or complex manual unrolling that would otherwise spike register pressure.

```metal
#include <metal_stdlib>
using namespace metal;

constexpr float CX[9] = {0.0f,  1.0f,  0.0f, -1.0f,  0.0f,  1.0f, -1.0f, -1.0f,  1.0f};
constexpr float CY[9] = {0.0f,  0.0f,  1.0f,  0.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f};
constexpr float W[9] = {
    4.0f / 9.0f,
    1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
    1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f
};

[[max_total_threads_per_threadgroup(256)]]
kernel void lbm_step(device const float *f_in   [[buffer(0)]],
                     device       float *f_out  [[buffer(1)]],
                     constant uint        &NX   [[buffer(2)]],
                     constant uint        &NY   [[buffer(3)]],
                     constant float       &tau  [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint j = gid.y;
    
    if (i >= NX || j >= NY) return;

    int inx = int(NX);
    int iny = int(NY);
    int N = inx * iny;
    int idx = int(j) * inx + int(i);

    // Fast 1D periodic boundary offset calculation
    int im1 = (i == 0) ? (inx - 1) : -1;
    int ip1 = (i == NX - 1) ? (1 - inx) : 1;
    int jm1 = (j == 0) ? (inx * (iny - 1)) : -inx;
    int jp1 = (j == NY - 1) ? -(inx * (iny - 1)) : inx;

    const int off[9] = {
        0,
        im1,
        jm1,
        ip1,
        jp1,
        im1 + jm1,
        ip1 + jm1,
        ip1 + jp1,
        im1 + jp1
    };

    float f[9];
    float rho = 0.0f;
    float ux = 0.0f;
    float uy = 0.0f;

    // Pull streaming and moments computation
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        float v = f_in[k * N + idx + off[k]];
        f[k] = v;
        rho += v;
        ux += CX[k] * v;
        uy += CY[k] * v;
    }

    float inv_rho = 1.0f / rho;
    ux *= inv_rho;
    uy *= inv_rho;

    float usq = ux * ux + uy * uy;
    float inv_tau = 1.0f / tau;
    float oma = 1.0f - inv_tau;
    
    float rho_inv_tau = rho * inv_tau;
    float term0 = 1.0f - 1.5f * usq;

    // BGK collision parameterized and heavily collapsed
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        float cu = CX[k] * ux + CY[k] * uy;
        float feq_inv_tau = W[k] * rho_inv_tau * (term0 + cu * (3.0f + 4.5f * cu));
        f_out[k * N + idx] = f[k] * oma + feq_inv_tau;
    }
}
```