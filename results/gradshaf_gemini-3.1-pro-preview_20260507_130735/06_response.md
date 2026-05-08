To optimize this compute kernel, two main improvements are applied:

1. **Threadgroup Memory Tiling in `gradshaf_step`**: Memory bandwidth is the primary bottleneck. The original step kernel fetched five cells from global memory per inner cell. By loading the domain tile (and its 1-cell halo) cooperatively into a `threadgroup` array, global memory reads are reduced from ~5 floats to slightly over 1 float per cell, massively shrinking memory traffic and directly unlocking bandwidth.
2. **Division-Free Float4 Vectorization in `gradshaf_axis_reduce`**: The incumbent performs integer division per vectorized memory load to map 1D indices back to 2D coordinates. This takes significant cycles in a tight loop. Instead of dividing each step, we pre-calculate the coordinate stride (`jump_i` and `jump_j`) matching the threadgroup step size. Tracking `i` and `j` natively via wrapping additions completely eradicates integer division latency from the inner loop.

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