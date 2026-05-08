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