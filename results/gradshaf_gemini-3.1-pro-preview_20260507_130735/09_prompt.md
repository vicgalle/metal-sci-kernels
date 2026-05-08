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
    uint interior_w = NR - 2;
    uint interior_h = NZ - 2;
    uint total_interior = interior_w * interior_h;
    
    float local_max = -INFINITY;
    uint idx = tid;
    
    if (idx < total_interior) {
        uint r = idx / interior_w;
        uint c = idx % interior_w;
        
        uint jump = tgsize;
        uint jump_r = jump / interior_w;
        uint jump_c = jump % interior_w;
        
        while (idx < total_interior) {
            local_max = max(local_max, psi[(r + 1) * NR + (c + 1)]);
            
            idx += jump;
            c += jump_c;
            r += jump_r;
            // Incremental 2D index wrapping to bypass modulo/division
            if (c >= interior_w) {
                c -= interior_w;
                r++;
            }
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
                          uint2 gid [[thread_position_in_grid]],
                          uint2 tid_in_tg [[thread_position_in_threadgroup]],
                          uint2 tsize [[threads_per_threadgroup]]) {
    
    uint i = gid.x;
    uint j = gid.y;
    
    uint tx = tid_in_tg.x;
    uint ty = tid_in_tg.y;
    
    bool valid = (i < NR && j < NZ);
    uint idx = valid ? (j * NR + i) : 0;
    
    float center = valid ? psi_in[idx] : 0.0f;
    
    // 18x18 tile perfectly accommodates up to 16x16 threadgroups + 1 cell halo bounds
    // The odd dimension ensures no memory bank conflicts.
    threadgroup float tile[18][18];
    threadgroup float tg_ax_inv;
    
    tile[ty + 1][tx + 1] = center;
    
    if (valid) {
        if (tx == 0 && i > 0) tile[ty + 1][0] = psi_in[idx - 1];
        if (tx == tsize.x - 1 && i < NR - 1) tile[ty + 1][tsize.x + 1] = psi_in[idx + 1];
        if (ty == 0 && j > 0) tile[0][tx + 1] = psi_in[idx - NR];
        if (ty == tsize.y - 1 && j < NZ - 1) tile[tsize.y + 1][tx + 1] = psi_in[idx + NR];
    }
    
    if (tx == 0 && ty == 0) {
        tg_ax_inv = 1.0f / psi_axis[0];
    }
    
    // Essential synchronization ensuring all halos are cached
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (!valid) return;
    
    bool is_boundary = (i == 0 || j == 0 || i == NR - 1 || j == NZ - 1);
    if (is_boundary) {
        psi_out[idx] = center;
        return;
    }
    
    float ax_inv = tg_ax_inv;
    
    // Computation completely unrolled into uniforms and FMAs
    float dr = dR;
    float dz = dZ;
    float inv_dr = 1.0f / dr;
    float inv_dz = 1.0f / dz;
    float inv_dr2 = inv_dr * inv_dr;
    float inv_dz2 = inv_dz * inv_dz;
    float a_C = -2.0f * (inv_dr2 + inv_dz2);
    float omega_inv_aC = omega / a_C;
    float half_inv_dr = 0.5f * inv_dr;
    float mu0_p_axis_4 = mu0 * p_axis * 4.0f;
    
    float R = Rmin + float(i) * dr;
    float term2 = half_inv_dr / R;
    
    float psi_norm = center * ax_inv;
    
    float rhs = 0.0f;
    if (psi_norm > 0.0f && psi_norm < 1.0f) {
        rhs = -mu0_p_axis_4 * (R * R) * psi_norm * (1.0f - psi_norm);
    }
    
    float psi_W = tile[ty + 1][tx];
    float psi_E = tile[ty + 1][tx + 2];
    float psi_N = tile[ty + 2][tx + 1];
    float psi_S = tile[ty][tx + 1];
    
    float delta_psi = a_C * center + inv_dr2 * (psi_W + psi_E) + term2 * (psi_W - psi_E) + inv_dz2 * (psi_N + psi_S);
    float r_val = rhs - delta_psi;
    
    psi_out[idx] = center + omega_inv_aC * r_val;
}
```

Result of previous attempt:
          65x65_30: correct, 0.75 ms, 2.0 GB/s (effective, 12 B/cell) (1.0% of 200 GB/s)
        257x257_40: correct, 3.22 ms, 9.8 GB/s (effective, 12 B/cell) (4.9% of 200 GB/s)
        513x513_30: correct, 10.38 ms, 9.1 GB/s (effective, 12 B/cell) (4.6% of 200 GB/s)
  score (gmean of fraction): 0.0284

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
    
    uint total4 = (NR * NZ) / 4;
    device const float4* psi4 = (device const float4*)psi;
    
    // Compute initial logical 2D coordinate for this thread's first element
    uint i = (tid * 4) % NR;
    uint j = (tid * 4) / NR;
    
    // Jump distance per loop iteration
    uint jump = tgsize * 4;
    uint jump_i = jump % NR;
    uint jump_j = jump / NR;
    
    uint idx = tid;
    while (idx < total4) {
        float4 val = psi4[idx];
        
        uint curr_i = i;
        uint curr_j = j;
        
        if (curr_i > 0 && curr_i < NR - 1 && curr_j > 0 && curr_j < NZ - 1) local_max = max(local_max, val[0]);
        curr_i++; if (curr_i == NR) { curr_i = 0; curr_j++; }
        
        if (curr_i > 0 && curr_i < NR - 1 && curr_j > 0 && curr_j < NZ - 1) local_max = max(local_max, val[1]);
        curr_i++; if (curr_i == NR) { curr_i = 0; curr_j++; }
        
        if (curr_i > 0 && curr_i < NR - 1 && curr_j > 0 && curr_j < NZ - 1) local_max = max(local_max, val[2]);
        curr_i++; if (curr_i == NR) { curr_i = 0; curr_j++; }
        
        if (curr_i > 0 && curr_i < NR - 1 && curr_j > 0 && curr_j < NZ - 1) local_max = max(local_max, val[3]);
        
        idx += tgsize;
        
        // Fast coordinate tracker bypassing division/modulo
        i += jump_i;
        j += jump_j;
        if (i >= NR) {
            i -= NR;
            j++;
        }
    }
    
    // Process scalar remainder if grid size is not a multiple of 4
    uint rem_start = total4 * 4;
    uint rem_k = rem_start + tid;
    if (rem_k < NR * NZ) {
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
    
    // Boundary cell fast path
    if (i == 0 || j == 0 || i == NR - 1 || j == NZ - 1) {
        psi_out[idx] = psi_in[idx];
        return;
    }
    
    float dr = dR;
    float dz = dZ;
    float inv_dr = 1.0f / dr;
    float inv_dz = 1.0f / dz;
    float inv_dr2 = inv_dr * inv_dr;
    float inv_dz2 = inv_dz * inv_dz;
    
    float a_C = -2.0f * (inv_dr2 + inv_dz2);
    float a_N = inv_dz2;
    
    float R = Rmin + float(i) * dr;
    float term2 = 0.5f * inv_dr / R;
    
    float a_W = inv_dr2 + term2;
    float a_E = inv_dr2 - term2;
    
    // Pre-cache central element and uniform reference point
    float psi_C = psi_in[idx];
    float ax = psi_axis[0];
    
    float psi_norm = psi_C / ax;
    
    float p_axis_4 = p_axis * 4.0f;
    float J_val = R * p_axis_4 * psi_norm * (1.0f - psi_norm);
    float J = (psi_norm > 0.0f && psi_norm < 1.0f) ? J_val : 0.0f;
    
    float rhs = -mu0 * R * J;
    
    // Direct cache hits, completely bypassing threadgroup synchronization
    float psi_W = psi_in[idx - 1];
    float psi_E = psi_in[idx + 1];
    float psi_N = psi_in[idx + NR];
    float psi_S = psi_in[idx - NR];
    
    float delta_psi = a_W * psi_W + a_E * psi_E + a_N * (psi_N + psi_S) + a_C * psi_C;
    float r = rhs - delta_psi;
    
    psi_out[idx] = psi_C + (omega / a_C) * r;
}
```

Incumbent result:
          65x65_30: correct, 0.32 ms, 4.8 GB/s (effective, 12 B/cell) (2.4% of 200 GB/s)
        257x257_40: correct, 1.88 ms, 16.8 GB/s (effective, 12 B/cell) (8.4% of 200 GB/s)
        513x513_30: correct, 3.23 ms, 29.3 GB/s (effective, 12 B/cell) (14.7% of 200 GB/s)
  score (gmean of fraction): 0.0664

## History

- iter  1: compile=OK | correct=True | score=0.044035250005550086
- iter  2: compile=OK | correct=True | score=0.018871323344846227
- iter  3: compile=OK | correct=True | score=0.04004499318144286
- iter  4: compile=OK | correct=True | score=0.036905301895395086
- iter  5: compile=OK | correct=True | score=0.017169134944783584
- iter  6: compile=OK | correct=True | score=0.052898508998715406
- iter  7: compile=OK | correct=True | score=0.06639526660306595
- iter  8: compile=OK | correct=True | score=0.028361572370522708

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
