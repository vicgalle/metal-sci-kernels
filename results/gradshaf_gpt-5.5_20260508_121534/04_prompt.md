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
        if (tid == 0u) {
            psi_axis[0] = -INFINITY;
        }
        return;
    }

    const uint NR_int = NR - 2u;
    const uint NZ_int = NZ - 2u;
    const uint total  = NR_int * NZ_int;

    float local_max = -INFINITY;

    if (tid < total) {
        // Initial flattened-interior coordinate for k = tid.
        uint i_int = tid % NR_int;
        uint j_int = tid / NR_int;
        uint idx   = (j_int + 1u) * NR + (i_int + 1u);

        // Advance by tgsize in flattened-interior space without hot-loop div/mod.
        const uint q = tgsize / NR_int;
        const uint r = tgsize - q * NR_int;
        const uint base_idx_step = q * NR + r;  // plus 2 when crossing an interior row

        while (j_int < NZ_int) {
            local_max = max(local_max, psi[idx]);

            uint ni = i_int + r;
            idx += base_idx_step;
            j_int += q;

            if (ni >= NR_int) {
                ni -= NR_int;
                ++j_int;
                idx += 2u; // skip right boundary of old row + left boundary of new row
            }
            i_int = ni;
        }
    }

    const float sg_max = simd_max(local_max);

    if ((tid & 31u) == 0u) {
        partial[tid >> 5] = sg_max;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32u) {
        const uint nsg = (tgsize + 31u) >> 5;
        float v = (tid < nsg) ? partial[tid] : -INFINITY;
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

    if (i >= NR || j >= NZ) {
        return;
    }

    const uint idx = j * NR + i;
    const float psi_C = psi_in[idx];

    if (i == 0u || j == 0u || i == NR - 1u || j == NZ - 1u) {
        psi_out[idx] = psi_C;
        return;
    }

    const float psi_W = psi_in[idx - 1u];
    const float psi_E = psi_in[idx + 1u];
    const float psi_S = psi_in[idx - NR];
    const float psi_N = psi_in[idx + NR];

    const float R = Rmin + float(i) * dR;
    const float axis = psi_axis[0];
    const float psi_norm = psi_C / axis;

    float shape = 0.0f;
    if (psi_norm > 0.0f && psi_norm < 1.0f) {
        shape = psi_norm * (1.0f - psi_norm);
    }

    float base;

    if (NR == NZ) {
        // For dR == dZ, the Jacobi center term cancels:
        // ψ_new(ω=1) = 1/4*(W+E+N+S) + dR/(8R)*(W-E)
        //              + μ0*p_axis*dR^2*R^2*ψn*(1-ψn).
        const float avg  = 0.25f * ((psi_W + psi_E) + (psi_N + psi_S));
        const float corr = (0.125f * dR / R) * (psi_W - psi_E);
        const float src  = (mu0 * p_axis) * (dR * dR) * (R * R) * shape;
        base = avg + corr + src;
    } else {
        // Generic algebraically-simplified form for non-square grids.
        const float inv_dR  = 1.0f / dR;
        const float inv_dZ  = 1.0f / dZ;
        const float inv_dR2 = inv_dR * inv_dR;
        const float inv_dZ2 = inv_dZ * inv_dZ;
        const float denom   = inv_dR2 + inv_dZ2;
        const float inv_2denom = 0.5f / denom;

        const float h_inv_RdR = 0.5f * inv_dR / R;

        base = inv_2denom *
               (inv_dR2 * (psi_W + psi_E) +
                inv_dZ2 * (psi_N + psi_S) +
                h_inv_RdR * (psi_W - psi_E));

        base += (2.0f * mu0 * p_axis / denom) * (R * R) * shape;
    }

    psi_out[idx] = psi_C + omega * (base - psi_C);
}
```

Result of previous attempt:
          65x65_30: correct, 0.72 ms, 2.1 GB/s (effective, 12 B/cell) (1.1% of 200 GB/s)
        257x257_40: correct, 4.08 ms, 7.8 GB/s (effective, 12 B/cell) (3.9% of 200 GB/s)
        513x513_30: correct, 10.92 ms, 8.7 GB/s (effective, 12 B/cell) (4.3% of 200 GB/s)
  score (gmean of fraction): 0.0261

## History

- iter  0: compile=OK | correct=True | score=0.022728115803763656
- iter  1: compile=OK | correct=True | score=0.017566168356858373
- iter  2: compile=OK | correct=True | score=0.017298549336246485
- iter  3: compile=OK | correct=True | score=0.026146912077090325

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
