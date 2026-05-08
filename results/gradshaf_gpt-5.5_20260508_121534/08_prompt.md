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

static inline float max4f(const float4 v) {
    return max(max(v.x, v.y), max(v.z, v.w));
}

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

        for (uint j = 1u + ly; j < row_stop; j += nsg) {
            const uint base = j * NR;
            uint i = 1u + (lx << 2u);

            for (; (i + 387u) < col_stop; i += 512u) {
                const float4 v0 = float4(*((device const packed_float4 *)(psi + base + i)));
                const float4 v1 = float4(*((device const packed_float4 *)(psi + base + i + 128u)));
                const float4 v2 = float4(*((device const packed_float4 *)(psi + base + i + 256u)));
                const float4 v3 = float4(*((device const packed_float4 *)(psi + base + i + 384u)));
                local_max = max(local_max, max(max4f(v0), max4f(v1)));
                local_max = max(local_max, max(max4f(v2), max4f(v3)));
            }

            for (; (i + 131u) < col_stop; i += 256u) {
                const float4 v0 = float4(*((device const packed_float4 *)(psi + base + i)));
                const float4 v1 = float4(*((device const packed_float4 *)(psi + base + i + 128u)));
                local_max = max(local_max, max(max4f(v0), max4f(v1)));
            }

            for (; (i + 3u) < col_stop; i += 128u) {
                const float4 v = float4(*((device const packed_float4 *)(psi + base + i)));
                local_max = max(local_max, max4f(v));
            }

            for (; i < col_stop; ++i) {
                local_max = max(local_max, psi[base + i]);
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
    threadgroup float vtile[18u * 16u];
    threadgroup float tg_asym[16u];
    threadgroup float tg_src[16u];
    threadgroup float tg_inv_axis;

    const uint i  = gid.x;
    const uint j  = gid.y;
    const uint lx = i & 15u;
    const uint ly = j & 15u;

    const bool valid = (i < NR) && (j < NZ);
    const uint idx = j * NR + i;

    const float psi_C = valid ? psi_in[idx] : 0.0f;

    vtile[(ly + 1u) * 16u + lx] = psi_C;

    if (ly == 0u) {
        vtile[lx] = (i < NR && j > 0u && j < NZ) ? psi_in[idx - NR] : 0.0f;

        if (i < NR) {
            const float R = fma(float(i), dR, Rmin);

            if (NR == NZ) {
                const float asym_coeff = 0.125f * dR;
                const float src_coeff  = (mu0 * p_axis) * (dR * dR);
                tg_asym[lx] = asym_coeff / R;
                tg_src[lx]  = src_coeff * (R * R);
            } else {
                const float inv_dR  = 1.0f / dR;
                const float inv_dZ  = 1.0f / dZ;
                const float inv_dR2 = inv_dR * inv_dR;
                const float inv_dZ2 = inv_dZ * inv_dZ;
                const float denom   = inv_dR2 + inv_dZ2;
                const float inv_2denom = 0.5f / denom;

                const float asym_coeff = inv_2denom * (0.5f * inv_dR);
                const float src_coeff  = (2.0f * mu0 * p_axis) / denom;
                tg_asym[lx] = asym_coeff / R;
                tg_src[lx]  = src_coeff * (R * R);
            }
        } else {
            tg_asym[lx] = 0.0f;
            tg_src[lx]  = 0.0f;
        }

        if (lx == 0u) {
            tg_inv_axis = 1.0f / psi_axis[0];
        }
    }

    if (ly == 15u) {
        vtile[17u * 16u + lx] = (i < NR && (j + 1u) < NZ) ? psi_in[idx + NR] : 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint ybit = j & 1u;
    const uint lane = (ybit << 4u) | lx;

    const ushort lane_w = ushort((lx > 0u)  ? (lane - 1u) : lane);
    const ushort lane_e = ushort((lx < 15u) ? (lane + 1u) : lane);

    const float sh_W = simd_shuffle(psi_C, lane_w);
    const float sh_E = simd_shuffle(psi_C, lane_e);

    if (!valid) {
        return;
    }

    if (i == 0u || j == 0u || (i + 1u) == NR || (j + 1u) == NZ) {
        psi_out[idx] = psi_C;
        return;
    }

    const float psi_W = (lx != 0u)  ? sh_W : psi_in[idx - 1u];
    const float psi_E = (lx != 15u) ? sh_E : psi_in[idx + 1u];
    const float psi_S = vtile[ly * 16u + lx];
    const float psi_N = vtile[(ly + 2u) * 16u + lx];

    float cR = 0.25f;
    float cZ = 0.25f;

    if (NR != NZ) {
        const float inv_dR  = 1.0f / dR;
        const float inv_dZ  = 1.0f / dZ;
        const float inv_dR2 = inv_dR * inv_dR;
        const float inv_dZ2 = inv_dZ * inv_dZ;
        const float denom   = inv_dR2 + inv_dZ2;
        const float inv_2denom = 0.5f / denom;

        cR = inv_2denom * inv_dR2;
        cZ = inv_2denom * inv_dZ2;
    }

    const float psi_norm = psi_C * tg_inv_axis;
    const float raw_shape = psi_norm * (1.0f - psi_norm);
    const float shape = ((psi_norm > 0.0f) && (psi_norm < 1.0f)) ? raw_shape : 0.0f;

    float base = fma(cR, (psi_W + psi_E), cZ * (psi_N + psi_S));
    base = fma(tg_asym[lx], (psi_W - psi_E), base);
    base = fma(tg_src[lx], shape, base);

    if (omega == 1.0f) {
        psi_out[idx] = base;
    } else {
        psi_out[idx] = fma(omega, (base - psi_C), psi_C);
    }
}
```

Result of previous attempt:
          65x65_30: correct, 0.98 ms, 1.6 GB/s (effective, 12 B/cell) (0.8% of 200 GB/s)
        257x257_40: correct, 2.40 ms, 13.2 GB/s (effective, 12 B/cell) (6.6% of 200 GB/s)
        513x513_30: correct, 3.94 ms, 24.0 GB/s (effective, 12 B/cell) (12.0% of 200 GB/s)
  score (gmean of fraction): 0.0395

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

static inline float max4f(const float4 v) {
    return max(max(v.x, v.y), max(v.z, v.w));
}

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

        for (uint j = 1u + ly; j < row_stop; j += nsg) {
            const uint base = j * NR;

            // Each SIMD lane owns 4 consecutive columns per vector chunk.
            // packed_float4 permits the naturally unaligned base+1 interior load.
            uint i = 1u + (lx << 2u);

            for (; (i + 3u) < col_stop; i += 128u) {
                const float4 v = float4(*((device const packed_float4 *)(psi + base + i)));
                local_max = max(local_max, max4f(v));
            }

            // Per-lane tail, at most 3 values for the final partial vector.
            for (; i < col_stop; ++i) {
                local_max = max(local_max, psi[base + i]);
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

    // With 16-wide threadgroups, each SIMDgroup covers two rows x 16 columns.
    const uint lx   = i & 15u;
    const uint ybit = j & 1u;
    const uint lane = (ybit << 4u) | lx;

    const ushort lane_w = ushort((lx > 0u)    ? (lane - 1u)  : lane);
    const ushort lane_e = ushort((lx < 15u)   ? (lane + 1u)  : lane);
    const ushort lane_s = ushort((ybit > 0u)  ? (lane - 16u) : lane);
    const ushort lane_n = ushort((ybit == 0u) ? (lane + 16u) : lane);

    const float sh_W = simd_shuffle(psi_C, lane_w);
    const float sh_E = simd_shuffle(psi_C, lane_e);
    const float sh_S = simd_shuffle(psi_C, lane_s);
    const float sh_N = simd_shuffle(psi_C, lane_n);

    // One reciprocal per SIMDgroup.
    const float inv_axis_src = (lane == 0u) ? (1.0f / psi_axis[0]) : 0.0f;
    const float inv_axis = simd_shuffle(inv_axis_src, ushort(0));

    float cR = 0.25f;
    float cZ = 0.25f;
    float asym;
    float src;

    if (NR == NZ) {
        const float asym_coeff = 0.125f * dR;
        const float src_coeff  = (mu0 * p_axis) * (dR * dR);

        float asym_src = 0.0f;
        float src_src  = 0.0f;

        // Broadcast column-dependent coefficients from the first row
        // of the 2-row SIMDgroup, avoiding duplicate R math in row 2.
        if (ybit == 0u) {
            const float R = fma(float(i), dR, Rmin);
            asym_src = asym_coeff / R;
            src_src  = src_coeff * (R * R);
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

        const float asym_coeff = inv_2denom * (0.5f * inv_dR);
        const float src_coeff  = (2.0f * mu0 * p_axis) / denom;

        float asym_src = 0.0f;
        float src_src  = 0.0f;

        if (ybit == 0u) {
            const float R = fma(float(i), dR, Rmin);
            asym_src = asym_coeff / R;
            src_src  = src_coeff * (R * R);
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

Incumbent result:
          65x65_30: correct, 0.49 ms, 3.1 GB/s (effective, 12 B/cell) (1.5% of 200 GB/s)
        257x257_40: correct, 3.05 ms, 10.4 GB/s (effective, 12 B/cell) (5.2% of 200 GB/s)
        513x513_30: correct, 4.46 ms, 21.2 GB/s (effective, 12 B/cell) (10.6% of 200 GB/s)
  score (gmean of fraction): 0.0439

## History

- iter  0: compile=OK | correct=True | score=0.022728115803763656
- iter  1: compile=OK | correct=True | score=0.017566168356858373
- iter  2: compile=OK | correct=True | score=0.017298549336246485
- iter  3: compile=OK | correct=True | score=0.026146912077090325
- iter  4: compile=OK | correct=True | score=0.035105332615358927
- iter  5: compile=OK | correct=True | score=0.039672840488318165
- iter  6: compile=OK | correct=True | score=0.043932040409169495
- iter  7: compile=OK | correct=True | score=0.039467091845384765

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
