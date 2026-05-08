## Task: gradshaf

Grad-Shafranov fixed-boundary equilibrium via K Picard outer steps. Per outer step:

  1. ψ_axis = max over INTERIOR of ψ      (i in [1, NR-1),                                            j in [1, NZ-1))
  2. For each interior (i, j):
       R         = Rmin + i*dR
       ψ_norm    = ψ[j,i] / ψ_axis
       J         = (0 < ψ_norm < 1) ? R * p_axis * 4 ψ_norm (1 − ψ_norm) : 0
       rhs       = −μ₀ * R * J
       Δ*ψ       = a_W ψ[j,i-1] + a_E ψ[j,i+1]
                 + a_N ψ[j+1,i] + a_S ψ[j-1,i] + a_C ψ[j,i]
         a_W = 1/dR² + 1/(2 R dR)     (R-dependent: 1/R term)
         a_E = 1/dR² − 1/(2 R dR)
         a_N = a_S = 1/dZ²
         a_C = −2/dR² − 2/dZ²
       r         = rhs − Δ*ψ
       ψ_new[j,i] = ψ[j,i] + ω * r / a_C
  3. Boundary cells (i==0, j==0, i==NR-1, j==NZ-1) MUST copy      ψ_in -> ψ_out unchanged (Dirichlet ψ=0 is preserved).

Storage is row-major float32 of shape (NZ, NR): linear index = j*NR + i, with i the fast (R) axis. Domain is fixed at R ∈ [1.0, 2.0], Z ∈ [-0.5, 0.5]; μ₀=1.0, p_axis=200.0, ω=1.0 are dimensionless and shared across all sizes. The host calls gradshaf_axis_reduce → gradshaf_step in alternation for K outer steps within one command buffer; psi_in/psi_out ping-pong each step. The reduction's output buffer (psi_axis) is a single-scalar device buffer that the host rebinds for each outer step. Effective DRAM traffic per outer step is ~12 B/cell (4 B reduction read + 8 B stencil read+write); the roofline is BW-bound.

## Required kernel signature(s)

```
kernel void gradshaf_axis_reduce(device const float *psi      [[buffer(0)]],
                                 device       float *psi_axis [[buffer(1)]],
                                 constant uint       &NR      [[buffer(2)]],
                                 constant uint       &NZ      [[buffer(3)]],
                                 uint tid                     [[thread_position_in_threadgroup]],
                                 uint tgsize                  [[threads_per_threadgroup]]);

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
                          uint2 gid [[thread_position_in_grid]]);

Reduce dispatch: 1-D, single threadgroup; the host picks `tgsize` (default 256) and dispatches `threadsPerGrid = tgsize`, `threadsPerThreadgroup = tgsize`. The single TG must reduce the entire interior into psi_axis[0]. You can swap to a multi-TG hierarchical reduction or simdgroup ops as long as psi_axis[0] holds the final max after one dispatch.

Step dispatch: 2-D, threadsPerGrid = (NR, NZ) rounded up to a multiple of (16, 16); threadsPerThreadgroup = (16, 16, 1) by default — guard with `if (i >= NR || j >= NZ) return;`. Boundary cells MUST copy psi_in -> psi_out unchanged. Each thread MUST update exactly one cell; the host will not shrink the dispatch.

IMPORTANT — threadgroup geometry is set by the host, not the kernel. The host always picks tg_w = 16 and only ever shrinks tg_h by halving (16×16 → 16×8 → 16×4 → 16×2 → 16×1) IF the kernel's [[max_total_threads_per_threadgroup(N)]] attribute forces a smaller cap. So the only TG shapes you can actually be dispatched with are (16, 16), (16, 8), (16, 4), (16, 2), (16, 1). You CANNOT get a (32, 8) or (8, 32) TG by writing the attribute or by `#define`-ing a tile size.

If you do threadgroup-memory tiling for the stencil, your tile dims MUST equal the dispatched TG dims (e.g. a 16×16 tile + halo, sized to the default TG). Computing a tile origin as `tgid.xy * TILE` only matches the dispatch when TILE equals the TG dims; otherwise tiles overlap (or leave gaps) and the result is non-deterministic / NaN. Same constraint for the reduction: its dispatched TG width is 256 threads (or smaller if you cap it via the max-threads attribute) — design your reduction around that.
```

## Your previous attempt

```metal
#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(256)]]
kernel void gradshaf_axis_reduce(device const float *psi      [[buffer(0)]],
                                 device       float *psi_axis [[buffer(1)]],
                                 constant uint       &NR      [[buffer(2)]],
                                 constant uint       &NZ      [[buffer(3)]],
                                 uint tid                     [[thread_position_in_threadgroup]],
                                 uint tgsize                  [[threads_per_threadgroup]]) {
    threadgroup float partial[8];

    if (NR <= 2u || NZ <= 2u) {
        if (tid == 0u) psi_axis[0] = -INFINITY;
        return;
    }

    const uint NRm1 = NR - 1u;
    const uint NZm1 = NZ - 1u;

    float local_max = -INFINITY;

    // Row-wise sweep of the interior avoids k / NR_int and k % NR_int
    // in the hot loop.  For the benchmark sizes and tgsize=256 this is
    // one or two coalesced loads per row per participating thread.
    for (uint j = 1u; j < NZm1; ++j) {
        const uint base = j * NR;

        uint i0 = tid + 1u;
        if (i0 < NRm1) {
            local_max = max(local_max, psi[base + i0]);
        }

        uint i1 = i0 + tgsize;
        if (i1 < NRm1) {
            local_max = max(local_max, psi[base + i1]);
        }

        for (uint i = i1 + tgsize; i < NRm1; i += tgsize) {
            local_max = max(local_max, psi[base + i]);
        }
    }

    // First reduce within each SIMDgroup, then reduce the SIMDgroup maxima
    // with the first SIMDgroup.  This replaces the 8-barrier tree reduction
    // with a single threadgroup barrier.
    float sg_max = simd_max(local_max);
    if ((tid & 31u) == 0u) {
        partial[tid >> 5] = sg_max;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32u) {
        const uint nsg = (tgsize + 31u) >> 5;
        float v = -INFINITY;
        if (tid < nsg) {
            v = partial[tid];
        }
        v = simd_max(v);
        if (tid == 0u) {
            psi_axis[0] = v;
        }
    }
}

[[max_total_threads_per_threadgroup(256)]]
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
    const uint i = gid.x;
    const uint j = gid.y;

    if (i >= NR || j >= NZ) return;

    const uint idx = j * NR + i;
    const float psi_C = psi_in[idx];

    // Host dispatches 16x16 threadgroups.  With SIMD width 32, each SIMDgroup
    // covers two adjacent 16-wide rows, so lane shuffles provide most E/W
    // neighbors and one of N/S without global loads.
    const uint lx = i & 15u;
    const uint ly = j & 15u;

    const float sh_W = simd_shuffle_up(psi_C, ushort(1));
    const float sh_E = simd_shuffle_down(psi_C, ushort(1));
    const float sh_S = simd_shuffle_up(psi_C, ushort(16));
    const float sh_N = simd_shuffle_down(psi_C, ushort(16));

    if (i == 0u || j == 0u || i == NR - 1u || j == NZ - 1u) {
        psi_out[idx] = psi_C;
        return;
    }

    float psi_W;
    if (lx == 0u) {
        psi_W = psi_in[idx - 1u];
    } else {
        psi_W = sh_W;
    }

    float psi_E;
    if (lx == 15u) {
        psi_E = psi_in[idx + 1u];
    } else {
        psi_E = sh_E;
    }

    float psi_N;
    float psi_S;
    if ((ly & 1u) == 0u) {
        psi_N = sh_N;
        psi_S = psi_in[idx - NR];
    } else {
        psi_N = psi_in[idx + NR];
        psi_S = sh_S;
    }

    const float R = Rmin + float(i) * dR;

    // Fixed domains have unit extent in both R and Z:
    // dR = 1/(NR-1), dZ = 1/(NZ-1).
    const float inv_dR  = float(NR - 1u);
    const float inv_dZ  = float(NZ - 1u);
    const float inv_dR2 = inv_dR * inv_dR;
    const float inv_dZ2 = inv_dZ * inv_dZ;

    const float h_inv_RdR = (0.5f * inv_dR) / R;
    const float a_W = inv_dR2 + h_inv_RdR;
    const float a_E = inv_dR2 - h_inv_RdR;
    const float a_N = inv_dZ2;
    const float a_S = inv_dZ2;
    const float a_C = -2.0f * inv_dR2 - 2.0f * inv_dZ2;
    const float inv_a_C = 1.0f / a_C;

    const float ax = psi_axis[0];
    const float psi_norm = psi_C / ax;

    float J = 0.0f;
    if (psi_norm > 0.0f && psi_norm < 1.0f) {
        J = R * p_axis * 4.0f * psi_norm * (1.0f - psi_norm);
    }

    const float rhs = -mu0 * R * J;

    const float delta_psi = a_W * psi_W + a_E * psi_E
                          + a_N * psi_N + a_S * psi_S
                          + a_C * psi_C;

    const float r = rhs - delta_psi;
    psi_out[idx] = psi_C + omega * r * inv_a_C;
}
```

Result of previous attempt:
          65x65_30: correct, 1.97 ms, 0.8 GB/s (effective, 12 B/cell) (0.4% of 200 GB/s)
        257x257_40: correct, 4.76 ms, 6.7 GB/s (effective, 12 B/cell) (3.3% of 200 GB/s)
        513x513_30: correct, 11.25 ms, 8.4 GB/s (effective, 12 B/cell) (4.2% of 200 GB/s)
  score (gmean of fraction): 0.0176

## Current best (incumbent)

```metal
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
```

Incumbent result:
          65x65_30: correct, 0.92 ms, 1.6 GB/s (effective, 12 B/cell) (0.8% of 200 GB/s)
        257x257_40: correct, 4.31 ms, 7.4 GB/s (effective, 12 B/cell) (3.7% of 200 GB/s)
        513x513_30: correct, 12.21 ms, 7.8 GB/s (effective, 12 B/cell) (3.9% of 200 GB/s)
  score (gmean of fraction): 0.0227

## History

- iter  0: compile=OK | correct=True | score=0.022728115803763656
- iter  1: compile=OK | correct=True | score=0.017566168356858373

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
