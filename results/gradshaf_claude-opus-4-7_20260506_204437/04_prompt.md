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

    for (uint k = tid; k < total; k += tgsize) {
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

// Tile dims must equal the dispatched TG dims. Host always uses tg_w = 16 and
// tg_h ∈ {16,8,4,2,1}, capped by max_total_threads_per_threadgroup. With the
// 256 cap we get the default 16×16 tile + 1-cell halo => 18×18 = 324 floats.
#define TILE_W 16
#define TILE_H 16
#define HALO_W (TILE_W + 2)
#define HALO_H (TILE_H + 2)

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
                          uint2 tgid [[threadgroup_position_in_grid]],
                          uint2 ltid [[thread_position_in_threadgroup]],
                          uint2 tgsz [[threads_per_threadgroup]]) {
    threadgroup float tile[HALO_H][HALO_W];

    // Origin of this tile in global coords (must use tgsz, since tg_h may be < TILE_H)
    int tile_i0 = int(tgid.x) * int(tgsz.x);
    int tile_j0 = int(tgid.y) * int(tgsz.y);

    uint lx = ltid.x;
    uint ly = ltid.y;
    uint lin = ly * tgsz.x + lx;
    uint nthreads = tgsz.x * tgsz.y;

    int NRi = int(NR);
    int NZi = int(NZ);

    // Cooperative halo load: fill (tgsz.y + 2) x (tgsz.x + 2) region.
    uint halo_w = tgsz.x + 2u;
    uint halo_h = tgsz.y + 2u;
    uint halo_total = halo_w * halo_h;

    for (uint k = lin; k < halo_total; k += nthreads) {
        uint hy = k / halo_w;
        uint hx = k - hy * halo_w;
        int gx = tile_i0 + int(hx) - 1;
        int gy = tile_j0 + int(hy) - 1;
        gx = clamp(gx, 0, NRi - 1);
        gy = clamp(gy, 0, NZi - 1);
        tile[hy][hx] = psi_in[gy * NRi + gx];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint i = gid.x;
    uint j = gid.y;
    if (i >= NR || j >= NZ) return;

    uint idx = j * NR + i;

    if (i == 0u || j == 0u || i == NR - 1u || j == NZ - 1u) {
        // Boundary: passthrough. Use tile load to avoid extra device fetch.
        psi_out[idx] = tile[ly + 1u][lx + 1u];
        return;
    }

    float psi_C = tile[ly + 1u][lx + 1u];
    float psi_W = tile[ly + 1u][lx     ];
    float psi_E = tile[ly + 1u][lx + 2u];
    float psi_S = tile[ly     ][lx + 1u];
    float psi_N = tile[ly + 2u][lx + 1u];

    float R         = Rmin + float(i) * dR;
    float inv_dR    = 1.0f / dR;
    float inv_dR2   = inv_dR * inv_dR;
    float inv_dZ2   = 1.0f / (dZ * dZ);
    float h_inv_RdR = 0.5f * inv_dR / R;
    float a_W  = inv_dR2 + h_inv_RdR;
    float a_E  = inv_dR2 - h_inv_RdR;
    float a_NS = inv_dZ2;
    float a_C  = -2.0f * inv_dR2 - 2.0f * inv_dZ2;
    float inv_aC = 1.0f / a_C;

    float ax = psi_axis[0];
    float psi_norm = psi_C / ax;
    float J = 0.0f;
    if (psi_norm > 0.0f && psi_norm < 1.0f) {
        J = R * p_axis * 4.0f * psi_norm * (1.0f - psi_norm);
    }
    float rhs = -mu0 * R * J;

    float delta_psi = a_W * psi_W + a_E * psi_E
                    + a_NS * (psi_N + psi_S)
                    + a_C  * psi_C;
    float r = rhs - delta_psi;
    psi_out[idx] = psi_C + omega * r * inv_aC;
}
```

Result of previous attempt:
          65x65_30: correct, 0.89 ms, 1.7 GB/s (effective, 12 B/cell) (0.9% of 200 GB/s)
        257x257_40: correct, 4.54 ms, 7.0 GB/s (effective, 12 B/cell) (3.5% of 200 GB/s)
        513x513_30: correct, 12.47 ms, 7.6 GB/s (effective, 12 B/cell) (3.8% of 200 GB/s)
  score (gmean of fraction): 0.0225

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

    // 2D strided sweep: each thread walks j with stride (tgsize / NR_int_chunk).
    // Simpler: linear strided sweep but compute (i,j) without modulo by stepping.
    uint total = NR_int * NZ_int;
    for (uint k = tid; k < total; k += tgsize) {
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

    // Boundary: copy through.
    if (i == 0u || j == 0u || i == NR - 1u || j == NZ - 1u) {
        psi_out[idx] = psi_in[idx];
        return;
    }

    // Direct device reads — rely on L1/L2 cache for the 5-point stencil reuse.
    float psi_C = psi_in[idx];
    float psi_W = psi_in[idx - 1u];
    float psi_E = psi_in[idx + 1u];
    float psi_S = psi_in[idx - NR];
    float psi_N = psi_in[idx + NR];

    float R         = Rmin + float(i) * dR;
    float inv_dR    = 1.0f / dR;
    float inv_dR2   = inv_dR * inv_dR;
    float inv_dZ2   = 1.0f / (dZ * dZ);
    float h_inv_RdR = 0.5f * inv_dR / R;
    float a_W = inv_dR2 + h_inv_RdR;
    float a_E = inv_dR2 - h_inv_RdR;
    float a_NS = inv_dZ2;
    float a_C = -2.0f * inv_dR2 - 2.0f * inv_dZ2;

    float ax = psi_axis[0];
    float psi_norm = psi_C / ax;
    float J = 0.0f;
    if (psi_norm > 0.0f && psi_norm < 1.0f) {
        J = R * p_axis * 4.0f * psi_norm * (1.0f - psi_norm);
    }
    float rhs = -mu0 * R * J;

    float delta_psi = a_W * psi_W + a_E * psi_E
                    + a_NS * (psi_N + psi_S)
                    + a_C * psi_C;
    float r = rhs - delta_psi;
    psi_out[idx] = psi_C + omega * r / a_C;
}
```

Incumbent result:
          65x65_30: correct, 0.83 ms, 1.8 GB/s (effective, 12 B/cell) (0.9% of 200 GB/s)
        257x257_40: correct, 4.53 ms, 7.0 GB/s (effective, 12 B/cell) (3.5% of 200 GB/s)
        513x513_30: correct, 12.40 ms, 7.6 GB/s (effective, 12 B/cell) (3.8% of 200 GB/s)
  score (gmean of fraction): 0.0231

## History

- iter  0: compile=OK | correct=True | score=0.02168492848331531
- iter  1: compile=OK | correct=True | score=0.022536772850618593
- iter  2: compile=OK | correct=True | score=0.02308977978685442
- iter  3: compile=OK | correct=True | score=0.02248367848436624

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
