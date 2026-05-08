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