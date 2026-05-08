// Naive seed for the Grad-Shafranov fixed-boundary Picard-Jacobi task.
//
// Two kernels per Picard outer step:
//   (1) gradshaf_axis_reduce — max-reduction over the interior of psi_in
//       into a single scalar buffer psi_axis[0]. Single-threadgroup naive
//       implementation: one TG of THREADS threads strided-sweeps the
//       interior cells, computes per-thread max, then a tree reduction in
//       threadgroup memory.
//
//   (2) gradshaf_step — per-cell stencil + nonlinear source. With the
//       Dirichlet ψ=0 boundary, ψ_norm = ψ/ψ_axis. The source is
//           J(R, ψ_norm) = R · p_axis · 4 · ψ_norm · (1 − ψ_norm)
//       masked to zero outside (0, 1). The Δ* discretization is a 5-point
//       stencil with R-dependent east/west weights:
//           a_W = 1/dR² + 1/(2 R dR)
//           a_E = 1/dR² − 1/(2 R dR)
//           a_N = a_S = 1/dZ²
//           a_C = −2/dR² − 2/dZ²
//       Jacobi update with under-relaxation:
//           ψ_new = ψ + ω · ( -μ₀ R J − Δ*ψ ) / a_C
//       Boundary cells (i==0, j==0, i==NR-1, j==NZ-1) copy through.
//
// Buffer layout (row-major, float32):
//   buffer 0 (in)  : psi_in     [NR * NZ]   psi[j*NR + i]
//   buffer 1 (out) : psi_out    [NR * NZ]   (gradshaf_step) OR
//                    psi_axis   [1]         (gradshaf_axis_reduce)
//   buffer 2       : psi_axis (gradshaf_step only) [1]
//   constants follow; see per-kernel signatures below.

#include <metal_stdlib>
using namespace metal;

// Naive single-threadgroup max-reduction over the interior cells of psi.
// Dispatch with grid = THREADS, threadgroup = THREADS (1 TG total).
[[max_total_threads_per_threadgroup(256)]]
kernel void gradshaf_axis_reduce(device const float *psi      [[buffer(0)]],
                                 device       float *psi_axis [[buffer(1)]],
                                 constant uint       &NR      [[buffer(2)]],
                                 constant uint       &NZ      [[buffer(3)]],
                                 uint tid                     [[thread_position_in_threadgroup]],
                                 uint tgsize                  [[threads_per_threadgroup]]) {
    threadgroup float partial[256];

    float local_max = -INFINITY;
    uint NR_int = NR - 2u;
    uint NZ_int = NZ - 2u;
    uint total  = NR_int * NZ_int;
    for (uint k = tid; k < total; k += tgsize) {
        uint i_int = k % NR_int;
        uint j_int = k / NR_int;
        uint i = i_int + 1u;
        uint j = j_int + 1u;
        float v = psi[j * NR + i];
        local_max = max(local_max, v);
    }
    partial[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction over threadgroup. Assumes tgsize is a power of 2.
    for (uint s = tgsize >> 1; s > 0u; s >>= 1) {
        if (tid < s) {
            partial[tid] = max(partial[tid], partial[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        psi_axis[0] = partial[0];
    }
}

kernel void gradshaf_step(device const float *psi_in   [[buffer(0)]],
                          device       float *psi_out  [[buffer(1)]],
                          device const float *psi_axis [[buffer(2)]],
                          constant uint       &NR      [[buffer(3)]],
                          constant uint       &NZ      [[buffer(4)]],
                          constant float      &Rmin    [[buffer(5)]],
                          constant float      &dR      [[buffer(6)]],
                          constant float      &dZ      [[buffer(7)]],
                          constant float      &p_axis  [[buffer(8)]],
                          constant float      &mu0     [[buffer(9)]],
                          constant float      &omega   [[buffer(10)]],
                          uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;  // R index (fast)
    uint j = gid.y;  // Z index (slow)
    if (i >= NR || j >= NZ) return;

    if (i == 0u || j == 0u || i == NR - 1u || j == NZ - 1u) {
        // Dirichlet: pass boundary values through unchanged.
        psi_out[j * NR + i] = psi_in[j * NR + i];
        return;
    }

    float R         = Rmin + float(i) * dR;
    float inv_dR2   = 1.0f / (dR * dR);
    float inv_dZ2   = 1.0f / (dZ * dZ);
    float h_inv_RdR = 0.5f / (R * dR);
    float a_W = inv_dR2 + h_inv_RdR;
    float a_E = inv_dR2 - h_inv_RdR;
    float a_N = inv_dZ2;
    float a_S = inv_dZ2;
    float a_C = -2.0f * inv_dR2 - 2.0f * inv_dZ2;

    float psi_C = psi_in[j * NR + i];
    float psi_W = psi_in[j * NR + (i - 1u)];
    float psi_E = psi_in[j * NR + (i + 1u)];
    float psi_N = psi_in[(j + 1u) * NR + i];
    float psi_S = psi_in[(j - 1u) * NR + i];

    float ax = psi_axis[0];
    float psi_norm = psi_C / ax;
    float J = 0.0f;
    if (psi_norm > 0.0f && psi_norm < 1.0f) {
        J = R * p_axis * 4.0f * psi_norm * (1.0f - psi_norm);
    }
    float rhs = -mu0 * R * J;

    float delta_psi = a_W * psi_W + a_E * psi_E
                    + a_N * psi_N + a_S * psi_S
                    + a_C * psi_C;
    float r = rhs - delta_psi;
    psi_out[j * NR + i] = psi_C + omega * r / a_C;
}
