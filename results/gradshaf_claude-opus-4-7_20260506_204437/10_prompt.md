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
    threadgroup float simd_partials[8];

    float local_max = -INFINITY;
    uint NR_int = NR - 2u;
    uint NZ_int = NZ - 2u;
    uint total = NR_int * NZ_int;

    uint k = tid;
    uint stride = tgsize;
    uint stride4 = stride * 4u;
    for (; k + stride4 <= total; k += stride4) {
        uint k0 = k;
        uint k1 = k + stride;
        uint k2 = k + 2u * stride;
        uint k3 = k + 3u * stride;
        uint j0 = k0 / NR_int, i0 = k0 - j0 * NR_int;
        uint j1 = k1 / NR_int, i1 = k1 - j1 * NR_int;
        uint j2 = k2 / NR_int, i2 = k2 - j2 * NR_int;
        uint j3 = k3 / NR_int, i3 = k3 - j3 * NR_int;
        float v0 = psi[(j0 + 1u) * NR + (i0 + 1u)];
        float v1 = psi[(j1 + 1u) * NR + (i1 + 1u)];
        float v2 = psi[(j2 + 1u) * NR + (i2 + 1u)];
        float v3 = psi[(j3 + 1u) * NR + (i3 + 1u)];
        local_max = max(local_max, max(max(v0, v1), max(v2, v3)));
    }
    for (; k < total; k += stride) {
        uint j_int = k / NR_int;
        uint i_int = k - j_int * NR_int;
        float v = psi[(j_int + 1u) * NR + (i_int + 1u)];
        local_max = max(local_max, v);
    }

    float sg_max = simd_max(local_max);
    uint sg_lane = tid & 31u;
    uint sg_id   = tid >> 5;
    if (sg_lane == 0u) {
        simd_partials[sg_id] = sg_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_id == 0u) {
        uint num_sgs = (tgsize + 31u) >> 5;
        float v = (sg_lane < num_sgs) ? simd_partials[sg_lane] : -INFINITY;
        v = simd_max(v);
        if (sg_lane == 0u) {
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
                          uint2 gid  [[thread_position_in_grid]],
                          uint  tid_in_tg [[thread_index_in_threadgroup]]) {
    // Broadcast psi_axis[0] via threadgroup memory: one device load per TG instead of one per thread.
    threadgroup float ax_shared;
    if (tid_in_tg == 0u) {
        ax_shared = psi_axis[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint i = gid.x;
    uint j = gid.y;
    if (i >= NR || j >= NZ) return;

    uint idx = j * NR + i;

    bool is_boundary = (i == 0u) | (j == 0u) | (i == NR - 1u) | (j == NZ - 1u);

    if (is_boundary) {
        psi_out[idx] = psi_in[idx];
        return;
    }

    // 5-point stencil — rely on L1/L2 cache for neighbor reuse.
    float psi_C = psi_in[idx];
    float psi_W = psi_in[idx - 1u];
    float psi_E = psi_in[idx + 1u];
    float psi_S = psi_in[idx - NR];
    float psi_N = psi_in[idx + NR];

    // Geometric coefficients
    float inv_dR    = 1.0f / dR;
    float inv_dR2   = inv_dR * inv_dR;
    float inv_dZ2   = 1.0f / (dZ * dZ);
    float R         = fma(float(i), dR, Rmin);
    float inv_R     = 1.0f / R;
    float h_inv_RdR = 0.5f * inv_dR * inv_R;
    float a_C       = -2.0f * (inv_dR2 + inv_dZ2);
    float inv_aC    = 1.0f / a_C;

    // Branchless J = R * p_axis * 4 * ψn * (1 - ψn) clamped to ψn ∈ (0,1)
    float ax       = ax_shared;
    float inv_ax   = 1.0f / ax;
    float psi_norm = psi_C * inv_ax;
    float pn_c     = clamp(psi_norm, 0.0f, 1.0f);
    float J        = R * p_axis * 4.0f * pn_c * (1.0f - pn_c);
    // Zero out exactly at endpoints to match spec (0 < ψn < 1)
    bool active    = (psi_norm > 0.0f) & (psi_norm < 1.0f);
    J              = active ? J : 0.0f;
    float rhs      = -mu0 * R * J;

    // Δ*ψ refactored: groups symmetric pairs to reduce dep chain
    //   = inv_dR2*(W+E) + inv_dZ2*(N+S) + h_inv_RdR*(W-E) + a_C*psi_C
    float we_sum   = psi_W + psi_E;
    float ns_sum   = psi_N + psi_S;
    float we_diff  = psi_W - psi_E;
    float delta_psi = fma(inv_dR2, we_sum,
                       fma(inv_dZ2, ns_sum,
                        fma(h_inv_RdR, we_diff,
                          a_C * psi_C)));
    float r = rhs - delta_psi;
    psi_out[idx] = fma(omega * r, inv_aC, psi_C);
}
```

Result of previous attempt:
          65x65_30: correct, 0.69 ms, 2.2 GB/s (effective, 12 B/cell) (1.1% of 200 GB/s)
        257x257_40: correct, 2.69 ms, 11.8 GB/s (effective, 12 B/cell) (5.9% of 200 GB/s)
        513x513_30: correct, 4.49 ms, 21.1 GB/s (effective, 12 B/cell) (10.6% of 200 GB/s)
  score (gmean of fraction): 0.0410

## Current best (incumbent)

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
    threadgroup float simd_partials[8];

    float local_max = -INFINITY;
    uint NR_int = NR - 2u;
    uint NZ_int = NZ - 2u;
    uint total = NR_int * NZ_int;

    // Unroll by 4 to better hide latency.
    uint k = tid;
    uint stride = tgsize;
    uint stride4 = stride * 4u;
    for (; k + stride4 <= total; k += stride4) {
        uint k0 = k;
        uint k1 = k + stride;
        uint k2 = k + 2u * stride;
        uint k3 = k + 3u * stride;
        uint j0 = k0 / NR_int, i0 = k0 - j0 * NR_int;
        uint j1 = k1 / NR_int, i1 = k1 - j1 * NR_int;
        uint j2 = k2 / NR_int, i2 = k2 - j2 * NR_int;
        uint j3 = k3 / NR_int, i3 = k3 - j3 * NR_int;
        float v0 = psi[(j0 + 1u) * NR + (i0 + 1u)];
        float v1 = psi[(j1 + 1u) * NR + (i1 + 1u)];
        float v2 = psi[(j2 + 1u) * NR + (i2 + 1u)];
        float v3 = psi[(j3 + 1u) * NR + (i3 + 1u)];
        local_max = max(local_max, max(max(v0, v1), max(v2, v3)));
    }
    for (; k < total; k += stride) {
        uint j_int = k / NR_int;
        uint i_int = k - j_int * NR_int;
        float v = psi[(j_int + 1u) * NR + (i_int + 1u)];
        local_max = max(local_max, v);
    }

    float sg_max = simd_max(local_max);
    uint sg_lane = tid & 31u;
    uint sg_id   = tid >> 5;
    if (sg_lane == 0u) {
        simd_partials[sg_id] = sg_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_id == 0u) {
        uint num_sgs = (tgsize + 31u) >> 5;
        float v = (sg_lane < num_sgs) ? simd_partials[sg_lane] : -INFINITY;
        v = simd_max(v);
        if (sg_lane == 0u) {
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
    uint i = gid.x;
    uint j = gid.y;
    if (i >= NR || j >= NZ) return;

    uint idx = j * NR + i;

    bool is_boundary = (i == 0u) | (j == 0u) | (i == NR - 1u) | (j == NZ - 1u);

    if (is_boundary) {
        psi_out[idx] = psi_in[idx];
        return;
    }

    // 5-point stencil reads — rely on L1/L2 cache.
    float psi_C = psi_in[idx];
    float psi_W = psi_in[idx - 1u];
    float psi_E = psi_in[idx + 1u];
    float psi_S = psi_in[idx - NR];
    float psi_N = psi_in[idx + NR];

    // Hoisted constants (compiler should fold across threads but explicit FMA-friendly form helps).
    float inv_dR    = 1.0f / dR;
    float inv_dR2   = inv_dR * inv_dR;
    float inv_dZ2   = 1.0f / (dZ * dZ);
    float R         = fma(float(i), dR, Rmin);
    float inv_R     = 1.0f / R;
    float h_inv_RdR = 0.5f * inv_dR * inv_R;
    float a_W = inv_dR2 + h_inv_RdR;
    float a_E = inv_dR2 - h_inv_RdR;
    float a_NS = inv_dZ2;
    float a_C = -2.0f * (inv_dR2 + inv_dZ2);
    float inv_aC = 1.0f / a_C;

    float ax = psi_axis[0];
    float psi_norm = psi_C / ax;
    float J = 0.0f;
    if (psi_norm > 0.0f && psi_norm < 1.0f) {
        J = R * p_axis * 4.0f * psi_norm * (1.0f - psi_norm);
    }
    float rhs = -mu0 * R * J;

    // Δ*ψ via FMAs
    float delta_psi = fma(a_W, psi_W,
                       fma(a_E, psi_E,
                        fma(a_NS, psi_N + psi_S,
                          a_C * psi_C)));
    float r = rhs - delta_psi;
    psi_out[idx] = fma(omega * r, inv_aC, psi_C);
}
```

Incumbent result:
          65x65_30: correct, 0.68 ms, 2.2 GB/s (effective, 12 B/cell) (1.1% of 200 GB/s)
        257x257_40: correct, 2.72 ms, 11.7 GB/s (effective, 12 B/cell) (5.8% of 200 GB/s)
        513x513_30: correct, 4.44 ms, 21.4 GB/s (effective, 12 B/cell) (10.7% of 200 GB/s)
  score (gmean of fraction): 0.0410

## History

- iter  2: compile=OK | correct=True | score=0.02308977978685442
- iter  3: compile=OK | correct=True | score=0.02248367848436624
- iter  4: compile=OK | correct=True | score=0.041040484659786436
- iter  5: compile=OK | correct=True | score=0.039697401230585856
- iter  6: compile=OK | correct=True | score=0.01632463120221179
- iter  7: compile=OK | correct=True | score=0.036005351032804805
- iter  8: compile=OK | correct=True | score=0.035071427732081675
- iter  9: compile=OK | correct=True | score=0.04102582970892118

## Stagnation notice

Your last 3 correct attempts all scored within 15% of
the incumbent without surpassing it. You are circling a local
optimum. STOP making incremental edits to the previous kernel.
Propose a STRUCTURALLY different approach — pick at least one of:

- Different memory layout (e.g. SoA <-> AoS; AA-pattern single-buffer
  in-place updates; pack scalars into a `float4` channel)
- Threadgroup-memory cooperative loads + `threadgroup_barrier` so a
  block of threads share one DRAM read across many compute steps
- SIMD-group cooperative reductions / shuffles via `simd_broadcast`,
  `simd_shuffle_*`, `simd_sum`, `simd_max`
- Multi-step / temporal kernel fusion: process several iterations
  per dispatch, holding state in threadgroup or register memory
- Persistent threadgroups (one threadgroup owns a tile across the
  entire run, eliminating per-step dispatch overhead)
- Mixed precision (fp16 inner loop with fp32 accumulation) where
  tolerance allows

A reworded version of the previous kernel will not break out of
this plateau.

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
