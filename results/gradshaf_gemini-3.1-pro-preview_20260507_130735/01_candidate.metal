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