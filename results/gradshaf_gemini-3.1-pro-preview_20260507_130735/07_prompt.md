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
    
    uint k_base = tid * 4;
    uint j = k_base / NR;
    uint i = k_base % NR;
    uint tg_step = tgsize * 4;
    
    // Precompute jump strides to avoid inner-loop integer division
    uint jump = tg_step - 4;
    uint jump_j = jump / NR;
    uint jump_i = jump % NR;
    
    while (k_base < total_cells4 * 4) {
        float4 val = psi4[k_base / 4];
        
        if (i > 0 && i < NR - 1 && j > 0 && j < NZ - 1) local_max = max(local_max, val[0]);
        i++; if (i == NR) { i = 0; j++; }
        
        if (i > 0 && i < NR - 1 && j > 0 && j < NZ - 1) local_max = max(local_max, val[1]);
        i++; if (i == NR) { i = 0; j++; }
        
        if (i > 0 && i < NR - 1 && j > 0 && j < NZ - 1) local_max = max(local_max, val[2]);
        i++; if (i == NR) { i = 0; j++; }
        
        if (i > 0 && i < NR - 1 && j > 0 && j < NZ - 1) local_max = max(local_max, val[3]);
        i++; if (i == NR) { i = 0; j++; }
        
        k_base += tg_step;
        i += jump_i;
        j += jump_j;
        if (i >= NR) {
            i -= NR;
            j++;
        }
    }
    
    uint rem_start = total_cells4 * 4;
    uint rem_k = rem_start + tid;
    if (rem_k < total_cells) {
        uint rem_j = rem_k / NR;
        uint rem_i = rem_k % NR;
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
                          uint2 gid [[thread_position_in_grid]],
                          uint2 lid [[thread_position_in_threadgroup]],
                          uint2 tgsize [[threads_per_threadgroup]]) {
    uint i = gid.x;
    uint j = gid.y;
    uint lx = lid.x;
    uint ly = lid.y;
    uint tx = tgsize.x;
    uint ty = tgsize.y;

    // Tile dims accommodate up to 16x16 dispatched Threadgroup size, plus a 1-element halo.
    threadgroup float tile[18][18];

    bool valid = (i < NR && j < NZ);
    
    // Load central cells
    if (valid) {
        tile[ly + 1][lx + 1] = psi_in[j * NR + i];
    }
    
    // Cooperatively load halos efficiently via boundary threads only
    if (lx == 0 && i > 0 && j < NZ) {
        tile[ly + 1][0] = psi_in[j * NR + i - 1];
    }
    if (lx == tx - 1 && i + 1 < NR && j < NZ) {
        tile[ly + 1][lx + 2] = psi_in[j * NR + i + 1];
    }
    if (ly == 0 && j > 0 && i < NR) {
        tile[0][lx + 1] = psi_in[(j - 1) * NR + i];
    }
    if (ly == ty - 1 && j + 1 < NZ && i < NR) {
        tile[ly + 2][lx + 1] = psi_in[(j + 1) * NR + i];
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (!valid) return;

    uint idx = j * NR + i;
    float psi_C = tile[ly + 1][lx + 1];
    
    // Boundary cell fast-path copy logic
    if (i == 0 || j == 0 || i == NR - 1 || j == NZ - 1) {
        psi_out[idx] = psi_C;
        return;
    }

    // Retrieve full stencil off threadgroup tile in exactly one memory cycle
    float psi_W = tile[ly + 1][lx];
    float psi_E = tile[ly + 1][lx + 2];
    float psi_S = tile[ly][lx + 1];
    float psi_N = tile[ly + 2][lx + 1];

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

Result of previous attempt:
          65x65_30: correct, 0.36 ms, 4.2 GB/s (effective, 12 B/cell) (2.1% of 200 GB/s)
        257x257_40: correct, 2.02 ms, 15.7 GB/s (effective, 12 B/cell) (7.9% of 200 GB/s)
        513x513_30: correct, 5.27 ms, 18.0 GB/s (effective, 12 B/cell) (9.0% of 200 GB/s)
  score (gmean of fraction): 0.0529

## History

- iter  0: compile=OK | correct=True | score=0.022996703224761085
- iter  1: compile=OK | correct=True | score=0.044035250005550086
- iter  2: compile=OK | correct=True | score=0.018871323344846227
- iter  3: compile=OK | correct=True | score=0.04004499318144286
- iter  4: compile=OK | correct=True | score=0.036905301895395086
- iter  5: compile=OK | correct=True | score=0.017169134944783584
- iter  6: compile=OK | correct=True | score=0.052898508998715406

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
