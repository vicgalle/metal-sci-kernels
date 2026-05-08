To maximize the memory bandwidth utilization, two major optimizations are applied:

1. **Fully Coalesced Division-Free Reduction:** The previous reduction kernel computed 2D indices per element, causing excessive ALU operations, and assigned non-contiguous chunks to threads, completely breaking memory coalescing. The new `gradshaf_axis_reduce` maps threads along the fast-varying axis `i` inside a `j` loop (`i += tgsize`), guaranteeing 100% coalesced scalar reads with zero integer division.
2. **SIMD-Shuffle Stencil (No Threadgroup Memory):** Instead of using threadgroup memory and barriers, `gradshaf_step` uses `simd_shuffle_up` and `simd_shuffle_down` to share the center load `psi_C` with adjacent threads. Because the host bounds `tg_w = 16` and SIMD width is 32, adjacent SIMD lanes perfectly correspond to adjacent memory cells in both X and Y directions. This eliminates ~60% of device memory/L1 cache load instructions per thread, avoiding all barrier overhead while keeping the ALU footprint minimal.

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
    uint nr = NR;
    uint nz = NZ;
    
    // Perfectly coalesced, division-free grid stride loop over the interior
    for (uint j = 1; j < nz - 1; j++) {
        uint row_start = j * nr;
        for (uint i = 1 + tid; i < nr - 1; i += tgsize) {
            local_max = max(local_max, psi[row_start + i]);
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
                          uint2 lid [[thread_position_in_threadgroup]]) {
    uint i = gid.x;
    uint j = gid.y;
    uint nr = NR;
    uint nz = NZ;

    // Do not return early so all threads participate in SIMD shuffles safely
    bool valid = (i < nr && j < nz);
    uint idx = valid ? (j * nr + i) : 0;
    
    float psi_C = valid ? psi_in[idx] : 0.0f;
    
    // Leverage SIMD subgroup width 32 and threadgroup width 16 for register-level data sharing
    float psi_W = simd_shuffle_up(psi_C, 1);
    if (lid.x == 0 && valid && i > 0) psi_W = psi_in[idx - 1];
    
    float psi_E = simd_shuffle_down(psi_C, 1);
    if (lid.x == 15 && valid && i + 1 < nr) psi_E = psi_in[idx + 1];
    
    float psi_S = simd_shuffle_up(psi_C, 16);
    if (lid.y % 2 == 0 && valid && j > 0) psi_S = psi_in[idx - nr];
    
    float psi_N = simd_shuffle_down(psi_C, 16);
    if (lid.y % 2 != 0 && valid && j + 1 < nz) psi_N = psi_in[idx + nr];
    
    if (!valid) return;
    
    if (i == 0 || j == 0 || i == nr - 1 || j == nz - 1) {
        psi_out[idx] = psi_C;
        return;
    }

    float dr_val = dR;
    float dz_val = dZ;
    float inv_dR = 1.0f / dr_val;
    float inv_dZ = 1.0f / dz_val;
    float inv_dR2 = inv_dR * inv_dR;
    float inv_dZ2 = inv_dZ * inv_dZ;
    float a_C = -2.0f * (inv_dR2 + inv_dZ2);
    float omega_inv_a_C = omega / a_C;
    
    float R = Rmin + float(i) * dr_val;
    float term2 = 0.5f * inv_dR / R;
    float a_W = inv_dR2 + term2;
    float a_E = inv_dR2 - term2;
    float a_N = inv_dZ2;

    float ax = psi_axis[0];
    float inv_ax = 1.0f / ax;
    float psi_norm = psi_C * inv_ax;
    
    float const_rhs = -mu0 * p_axis * 4.0f;
    float rhs_mult = const_rhs * R * R;
    
    bool valid_norm = (psi_norm > 0.0f && psi_norm < 1.0f);
    float rhs = valid_norm ? (rhs_mult * psi_norm * (1.0f - psi_norm)) : 0.0f;
    
    float delta_psi = a_W * psi_W + a_E * psi_E + a_N * (psi_N + psi_S) + a_C * psi_C;
    
    psi_out[idx] = psi_C + omega_inv_a_C * (rhs - delta_psi);
}
```