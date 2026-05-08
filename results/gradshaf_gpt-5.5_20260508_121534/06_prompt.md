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

    float local_max = -INFINITY;

    if (NR > 2u && NZ > 2u) {
        const uint lx       = tid & 31u;
        const uint ly       = tid >> 5;
        const uint nsg      = (tgsize + 31u) >> 5;
        const uint col_stop = NR - 1u;
        const uint row_stop = NZ - 1u;

        if (col_stop <= 128u) {
            // Small-width path: less remainder overhead.
            for (uint j = 1u + ly; j < row_stop; j += nsg) {
                const uint base = j * NR;
                uint i = 1u + lx;

                for (; (i + 32u) < col_stop; i += 64u) {
                    const float a = psi[base + i];
                    const float b = psi[base + i + 32u];
                    local_max = max(local_max, max(a, b));
                }
                if (i < col_stop) {
                    local_max = max(local_max, psi[base + i]);
                }
            }
        } else {
            // Wider rows: 8-way unroll to reduce loop overhead and expose ILP.
            for (uint j = 1u + ly; j < row_stop; j += nsg) {
                const uint base = j * NR;
                uint i = 1u + lx;

                for (; (i + 224u) < col_stop; i += 256u) {
                    const float v0 = psi[base + i];
                    const float v1 = psi[base + i + 32u];
                    const float v2 = psi[base + i + 64u];
                    const float v3 = psi[base + i + 96u];
                    const float v4 = psi[base + i + 128u];
                    const float v5 = psi[base + i + 160u];
                    const float v6 = psi[base + i + 192u];
                    const float v7 = psi[base + i + 224u];

                    const float m01 = max(v0, v1);
                    const float m23 = max(v2, v3);
                    const float m45 = max(v4, v5);
                    const float m67 = max(v6, v7);
                    local_max = max(local_max, max(max(m01, m23), max(m45, m67)));
                }

                if (i < col_stop)          local_max = max(local_max, psi[base + i]);
                if ((i + 32u) < col_stop)  local_max = max(local_max, psi[base + i + 32u]);
                if ((i + 64u) < col_stop)  local_max = max(local_max, psi[base + i + 64u]);
                if ((i + 96u) < col_stop)  local_max = max(local_max, psi[base + i + 96u]);
                if ((i + 128u) < col_stop) local_max = max(local_max, psi[base + i + 128u]);
                if ((i + 160u) < col_stop) local_max = max(local_max, psi[base + i + 160u]);
                if ((i + 192u) < col_stop) local_max = max(local_max, psi[base + i + 192u]);
                if ((i + 224u) < col_stop) local_max = max(local_max, psi[base + i + 224u]);
            }
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

    const bool valid = (i < NR) && (j < NZ);
    const uint idx = j * NR + i;

    const float psi_C = valid ? psi_in[idx] : 0.0f;

    // For a 16-wide TG, each SIMDgroup is two consecutive rows of 16 threads.
    const uint lx    = i & 15u;
    const uint ybit  = j & 1u;
    const uint lane  = (ybit << 4u) | lx;

    const ushort lane_w = ushort((lx   > 0u)  ? (lane - 1u)  : lane);
    const ushort lane_e = ushort((lx   < 15u) ? (lane + 1u)  : lane);
    const ushort lane_s = ushort((ybit > 0u)  ? (lane - 16u) : lane);
    const ushort lane_n = ushort((ybit == 0u) ? (lane + 16u) : lane);

    const float sh_W = simd_shuffle(psi_C, lane_w);
    const float sh_E = simd_shuffle(psi_C, lane_e);
    const float sh_S = simd_shuffle(psi_C, lane_s);
    const float sh_N = simd_shuffle(psi_C, lane_n);

    // One reciprocal of psi_axis per SIMDgroup.
    const float inv_axis_src = (lane == 0u) ? (1.0f / psi_axis[0]) : 0.0f;
    const float inv_axis = simd_shuffle(inv_axis_src, ushort(0));

    float asym = 0.0f;
    float src  = 0.0f;
    float cR   = 0.25f;
    float cZ   = 0.25f;

    const float R = Rmin + float(i) * dR;

    if (NR == NZ) {
        float asym_src = 0.0f;
        float src_src  = 0.0f;

        if (ybit == 0u) {
            asym_src = (0.125f * dR) / R;
            src_src  = (mu0 * p_axis) * (dR * dR) * (R * R);
        }

        asym = simd_shuffle(asym_src, ushort(lx));
        src  = simd_shuffle(src_src,  ushort(lx));
    } else {
        const float inv_dR  = 1.0f / dR;
        const float inv_dZ  = 1.0f / dZ;
        const float inv_dR2 = inv_dR * inv_dR;
        const float inv_dZ2 = inv_dZ * inv_dZ;
        const float denom   = inv_dR2 + inv_dZ2;
        const float inv_2denom = 0.5f / denom;

        cR = inv_2denom * inv_dR2;
        cZ = inv_2denom * inv_dZ2;

        float asym_src = 0.0f;
        float src_src  = 0.0f;

        if (ybit == 0u) {
            asym_src = inv_2denom * (0.5f * inv_dR / R);
            src_src  = (2.0f * mu0 * p_axis / denom) * (R * R);
        }

        asym = simd_shuffle(asym_src, ushort(lx));
        src  = simd_shuffle(src_src,  ushort(lx));
    }

    if (!valid) {
        return;
    }

    if (i == 0u || j == 0u || (i + 1u) == NR || (j + 1u) == NZ) {
        psi_out[idx] = psi_C;
        return;
    }

    const float psi_W = (lx != 0u)    ? sh_W : psi_in[idx - 1u];
    const float psi_E = (lx != 15u)   ? sh_E : psi_in[idx + 1u];
    const float psi_S = (ybit != 0u)  ? sh_S : psi_in[idx - NR];
    const float psi_N = (ybit == 0u)  ? sh_N : psi_in[idx + NR];

    const float psi_norm = psi_C * inv_axis;
    const float raw_shape = psi_norm * (1.0f - psi_norm);
    const float shape = ((psi_norm > 0.0f) && (psi_norm < 1.0f)) ? raw_shape : 0.0f;

    float base = fma(cR, (psi_W + psi_E), cZ * (psi_N + psi_S));
    base = fma(asym, (psi_W - psi_E), base);
    base = fma(src, shape, base);

    if (omega == 1.0f) {
        psi_out[idx] = base;
    } else {
        psi_out[idx] = fma(omega, (base - psi_C), psi_C);
    }
}
```

Result of previous attempt:
          65x65_30: correct, 0.76 ms, 2.0 GB/s (effective, 12 B/cell) (1.0% of 200 GB/s)
        257x257_40: correct, 2.87 ms, 11.0 GB/s (effective, 12 B/cell) (5.5% of 200 GB/s)
        513x513_30: correct, 4.20 ms, 22.5 GB/s (effective, 12 B/cell) (11.3% of 200 GB/s)
  score (gmean of fraction): 0.0397

## History

- iter  0: compile=OK | correct=True | score=0.022728115803763656
- iter  1: compile=OK | correct=True | score=0.017566168356858373
- iter  2: compile=OK | correct=True | score=0.017298549336246485
- iter  3: compile=OK | correct=True | score=0.026146912077090325
- iter  4: compile=OK | correct=True | score=0.035105332615358927
- iter  5: compile=OK | correct=True | score=0.039672840488318165

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
