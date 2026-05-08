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
    float local_max = -INFINITY;
    
    uint total_cells = NR * NZ;
    uint total_cells4 = total_cells / 4;
    device const float4* psi4 = (device const float4*)psi;
    
    uint k4 = tid;
    while (k4 < total_cells4) {
        float4 val = psi4[k4];
        uint k = k4 * 4;
        uint j = k / NR;
        uint i = k - j * NR;
        
        // Fast path: if the entire float4 block is strictly within the interior
        if (j > 0 && j < NZ - 1 && i > 0 && i + 3 < NR - 1) {
            local_max = max(local_max, max(max(val.x, val.y), max(val.z, val.w)));
        } else {
            // Slower fallback for boundary crossings
            uint temp_i = i;
            uint temp_j = j;
            for (int c = 0; c < 4; c++) {
                if (temp_j > 0 && temp_j < NZ - 1 && temp_i > 0 && temp_i < NR - 1) {
                    local_max = max(local_max, val[c]);
                }
                temp_i++;
                if (temp_i == NR) {
                    temp_i = 0;
                    temp_j++;
                }
            }
        }
        k4 += tgsize;
    }
    
    uint rem_start = total_cells4 * 4;
    for (uint rem_k = rem_start + tid; rem_k < total_cells; rem_k += tgsize) {
        uint rem_j = rem_k / NR;
        uint rem_i = rem_k - rem_j * NR;
        if (rem_j > 0 && rem_j < NZ - 1 && rem_i > 0 && rem_i < NR - 1) {
            local_max = max(local_max, psi[rem_k]);
        }
    }
    
    local_max = simd_max(local_max);
    
    if (tgsize <= 32) {
        if (tid == 0) psi_axis[0] = local_max;
        return;
    }
    
    threadgroup float shared_max[32];
    uint lane = tid % 32;
    uint warp = tid / 32;
    
    if (lane == 0) shared_max[warp] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (warp == 0) {
        uint num_warps = (tgsize + 31) / 32;
        float val = (tid < num_warps) ? shared_max[lane] : -INFINITY;
        val = simd_max(val);
        if (tid == 0) psi_axis[0] = val;
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
    float psi_C = psi_in[idx];
    
    // In Metal on Apple Silicon, warps are typically sized 32 and threadgroups are sweeping X fastest.
    // For a 16-width threadgroup, this yields a perfectly deterministic 16x2 SIMD group structure.
    uint tx = i % 16;
    uint ty_in_simd = j % 2;
    
    // Safely clamp edge memory addresses for threads skipping the SIMD shuffle on borders
    uint idx_E = min(idx + 1, NR * NZ - 1);
    uint idx_W = (idx > 0) ? idx - 1 : 0;
    uint idx_N = (idx + NR < NR * NZ) ? idx + NR : idx;
    uint idx_S = (idx >= NR) ? idx - NR : 0;
    
    // Cooperative register sharing: fetches ~3 fewer floats per cell
    float psi_W = (tx > 0) ? simd_shuffle_up(psi_C, 1) : psi_in[idx_W];
    float psi_E = (tx < 15) ? simd_shuffle_down(psi_C, 1) : psi_in[idx_E];
    
    float psi_N = (ty_in_simd == 0) ? simd_shuffle_down(psi_C, 16) : psi_in[idx_N];
    float psi_S = (ty_in_simd == 1) ? simd_shuffle_up(psi_C, 16)   : psi_in[idx_S];
    
    if (i == 0 || j == 0 || i == NR - 1 || j == NZ - 1) {
        psi_out[idx] = psi_C;
        return;
    }

    float inv_dR = 1.0f / dR;
    float inv_dZ = 1.0f / dZ;
    float inv_dR2 = inv_dR * inv_dR;
    float inv_dZ2 = inv_dZ * inv_dZ;
    float a_N = inv_dZ2;
    float a_C = -2.0f * (inv_dR2 + inv_dZ2);
    float omega_inv_a_C = omega / a_C;

    float R = Rmin + float(i) * dR;
    float term2 = (0.5f * inv_dR) / R;
    float a_W = inv_dR2 + term2;
    float a_E = inv_dR2 - term2;

    float psi_norm = psi_C / psi_axis[0];
    
    float J = 0.0f;
    if (psi_norm > 0.0f && psi_norm < 1.0f) {
        J = R * (p_axis * 4.0f) * psi_norm * (1.0f - psi_norm);
    }
    
    float rhs = -mu0 * R * J;
    float delta_psi = a_W * psi_W + a_E * psi_E + a_N * (psi_N + psi_S) + a_C * psi_C;
    
    psi_out[idx] = psi_C + omega_inv_a_C * (rhs - delta_psi);
}
```

Result of previous attempt:
          65x65_30: correct, 0.61 ms, 2.5 GB/s (effective, 12 B/cell) (1.2% of 200 GB/s)
        257x257_40: correct, 3.45 ms, 9.2 GB/s (effective, 12 B/cell) (4.6% of 200 GB/s)
        513x513_30: correct, 4.23 ms, 22.4 GB/s (effective, 12 B/cell) (11.2% of 200 GB/s)
  score (gmean of fraction): 0.0400

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
    float local_max = -INFINITY;
    
    uint total_cells = NR * NZ;
    uint total_cells4 = total_cells / 4;
    
    device const float4* psi4 = (device const float4*)psi;
    uint k4 = tid;
    
    while (k4 < total_cells4) {
        float4 val = psi4[k4];
        uint k_base = k4 * 4;
        uint j = k_base / NR;
        uint i = k_base % NR;
        
        for (int c = 0; c < 4; c++) {
            if (i > 0 && i < NR - 1 && j > 0 && j < NZ - 1) {
                local_max = max(local_max, val[c]);
            }
            i++; 
            if (i == NR) { 
                i = 0; 
                j++; 
            }
        }
        k4 += tgsize;
    }
    
    uint rem_start = total_cells4 * 4;
    uint rem_k = rem_start + tid;
    if (rem_k < total_cells) {
        uint rem_i = rem_k % NR;
        uint rem_j = rem_k / NR;
        if (rem_i > 0 && rem_i < NR - 1 && rem_j > 0 && rem_j < NZ - 1) {
            local_max = max(local_max, psi[rem_k]);
        }
    }
    
    local_max = simd_max(local_max);
    
    if (tgsize <= 32) {
        if (tid == 0) psi_axis[0] = local_max;
        return;
    }
    
    threadgroup float shared_max[32];
    uint lane = tid % 32;
    uint warp = tid / 32;
    
    if (lane == 0) shared_max[warp] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (warp == 0) {
        uint num_warps = (tgsize + 31) / 32;
        float val = (tid < num_warps) ? shared_max[lane] : -INFINITY;
        val = simd_max(val);
        if (tid == 0) psi_axis[0] = val;
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
    
    if (i == 0 || j == 0 || i == NR - 1 || j == NZ - 1) {
        psi_out[idx] = psi_in[idx];
        return;
    }

    float dr_val = dR;
    float dz_val = dZ;
    float inv_dR = 1.0f / dr_val;
    float inv_dZ = 1.0f / dz_val;
    float inv_dR2 = inv_dR * inv_dR;
    float inv_dZ2 = inv_dZ * inv_dZ;
    float a_N = inv_dZ2;
    float a_C = -2.0f * (inv_dR2 + inv_dZ2);
    float omega_inv_a_C = omega / a_C;
    float p_axis_4 = p_axis * 4.0f;
    float neg_mu0 = -mu0;

    float R = Rmin + float(i) * dr_val;
    float term2 = 0.5f * inv_dR / R;
    float a_W = inv_dR2 + term2;
    float a_E = inv_dR2 - term2;

    float psi_C = psi_in[idx];
    float psi_W = psi_in[idx - 1];
    float psi_E = psi_in[idx + 1];
    float psi_N = psi_in[idx + NR];
    float psi_S = psi_in[idx - NR];

    float ax = psi_axis[0];
    float inv_ax = 1.0f / ax;
    float psi_norm = psi_C * inv_ax;
    
    float J = 0.0f;
    if (psi_norm > 0.0f && psi_norm < 1.0f) {
        J = R * p_axis_4 * psi_norm * (1.0f - psi_norm);
    }
    
    float rhs = neg_mu0 * R * J;
    float delta_psi = a_W * psi_W + a_E * psi_E + a_N * (psi_N + psi_S) + a_C * psi_C;
    float r = rhs - delta_psi;
    
    psi_out[idx] = psi_C + omega_inv_a_C * r;
}
```

Incumbent result:
          65x65_30: correct, 0.54 ms, 2.8 GB/s (effective, 12 B/cell) (1.4% of 200 GB/s)
        257x257_40: correct, 3.17 ms, 10.0 GB/s (effective, 12 B/cell) (5.0% of 200 GB/s)
        513x513_30: correct, 3.89 ms, 24.3 GB/s (effective, 12 B/cell) (12.2% of 200 GB/s)
  score (gmean of fraction): 0.0440

## History

- iter  0: compile=OK | correct=True | score=0.022996703224761085
- iter  1: compile=OK | correct=True | score=0.044035250005550086
- iter  2: compile=OK | correct=True | score=0.018871323344846227
- iter  3: compile=OK | correct=True | score=0.04004499318144286

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
