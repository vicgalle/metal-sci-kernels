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