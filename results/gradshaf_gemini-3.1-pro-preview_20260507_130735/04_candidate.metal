#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(256)]]
kernel void gradshaf_axis_reduce(device const float *psi      [[buffer(0)]],
                                 device       float *psi_axis [[buffer(1)]],
                                 constant uint       &NR      [[buffer(2)]],
                                 constant uint       &NZ      [[buffer(3)]],
                                 uint tid                     [[thread_position_in_threadgroup]],
                                 uint tgsize                  [[threads_per_threadgroup]]) {
    uint total_cells = NR * NZ;
    
    // Assign contiguous blocks to threads to completely avoid division inside the inner loop
    uint chunk_size = (total_cells + tgsize - 1) / tgsize;
    chunk_size = ((chunk_size + 3) / 4) * 4; // Align to float4
    
    uint start_k = tid * chunk_size;
    uint end_k = min(start_k + chunk_size, total_cells);
    
    uint j = start_k / NR;
    uint i = start_k % NR;
    
    device const float4* psi4 = (device const float4*)psi;
    uint k = start_k;
    float local_max = -INFINITY;
    
    while (k + 3 < end_k) {
        float4 val = psi4[k / 4];
        if (i > 0 && i + 3 < NR - 1 && j > 0 && j < NZ - 1) {
            local_max = max(local_max, max(max(val.x, val.y), max(val.z, val.w)));
            i += 4;
        } else {
            for (int c = 0; c < 4; c++) {
                if (i > 0 && i < NR - 1 && j > 0 && j < NZ - 1) {
                    local_max = max(local_max, val[c]);
                }
                i++;
                if (i == NR) { i = 0; j++; }
            }
        }
        k += 4;
    }
    
    while (k < end_k) {
        if (i > 0 && i < NR - 1 && j > 0 && j < NZ - 1) {
            local_max = max(local_max, psi[k]);
        }
        i++;
        if (i == NR) { i = 0; j++; }
        k++;
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

    // Direct uniform mapping saves register spill/overhead
    float inv_dR = 1.0f / dR;
    float inv_dZ = 1.0f / dZ;
    float inv_dR2 = inv_dR * inv_dR;
    float inv_dZ2 = inv_dZ * inv_dZ;
    float a_C = -2.0f * (inv_dR2 + inv_dZ2);
    float omega_inv_a_C = omega / a_C;

    float R = Rmin + float(i) * dR;
    float term2 = 0.5f * inv_dR / R;
    float a_W = inv_dR2 + term2;
    float a_E = inv_dR2 - term2;

    // L1 cache easily absorbs a local 5-point stencil reading overlapping bounds natively 
    float psi_C = psi_in[idx];
    float psi_W = psi_in[idx - 1];
    float psi_E = psi_in[idx + 1];
    float psi_N = psi_in[idx + NR];
    float psi_S = psi_in[idx - NR];

    float inv_ax = 1.0f / psi_axis[0];
    float psi_norm = psi_C * inv_ax;
    
    float const_rhs = -mu0 * p_axis * 4.0f;
    bool valid = (psi_norm > 0.0f && psi_norm < 1.0f);
    
    // Conditionally selected float compiles directly to `csel`, preventing thread warp divergence
    float rhs = valid ? (const_rhs * R * R * psi_norm * (1.0f - psi_norm)) : 0.0f;
    
    // Extracted inv_dZ2 * (N + S) removes redundant multiply
    float delta_psi = a_W * psi_W + a_E * psi_E + inv_dZ2 * (psi_N + psi_S) + a_C * psi_C;
    
    psi_out[idx] = psi_C + omega_inv_a_C * (rhs - delta_psi);
}