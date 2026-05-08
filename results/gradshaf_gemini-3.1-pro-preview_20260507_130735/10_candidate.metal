#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(256)]]
kernel void gradshaf_axis_reduce(device const float * restrict psi      [[buffer(0)]],
                                 device       float * restrict psi_axis [[buffer(1)]],
                                 constant uint       &NR      [[buffer(2)]],
                                 constant uint       &NZ      [[buffer(3)]],
                                 uint tid                     [[thread_position_in_threadgroup]],
                                 uint tgsize                  [[threads_per_threadgroup]]) {
    uint interior_w = NR - 2;
    uint interior_h = NZ - 2;
    uint total_interior = interior_w * interior_h;
    
    // Tracks the exact 2D coordinates within the interior space (zero-based)
    uint i_in = tid % interior_w;
    uint j_in = tid / interior_w;
    
    // Jump distance per iteration avoids modulus operation inside the loop
    uint jump_i = tgsize % interior_w;
    uint jump_j = tgsize / interior_w;
    
    float local_max = -INFINITY;
    uint idx = tid;
    
    // Unrolled by 4 to minimize branch instructions
    while (idx + 3 * tgsize < total_interior) {
        local_max = max(local_max, psi[(j_in + 1) * NR + (i_in + 1)]);
        i_in += jump_i; j_in += jump_j;
        if (i_in >= interior_w) { i_in -= interior_w; j_in++; }
        
        local_max = max(local_max, psi[(j_in + 1) * NR + (i_in + 1)]);
        i_in += jump_i; j_in += jump_j;
        if (i_in >= interior_w) { i_in -= interior_w; j_in++; }
        
        local_max = max(local_max, psi[(j_in + 1) * NR + (i_in + 1)]);
        i_in += jump_i; j_in += jump_j;
        if (i_in >= interior_w) { i_in -= interior_w; j_in++; }
        
        local_max = max(local_max, psi[(j_in + 1) * NR + (i_in + 1)]);
        i_in += jump_i; j_in += jump_j;
        if (i_in >= interior_w) { i_in -= interior_w; j_in++; }
        
        idx += 4 * tgsize;
    }
    
    // Remainder handling
    while (idx < total_interior) {
        local_max = max(local_max, psi[(j_in + 1) * NR + (i_in + 1)]);
        i_in += jump_i; j_in += jump_j;
        if (i_in >= interior_w) { i_in -= interior_w; j_in++; }
        idx += tgsize;
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
kernel void gradshaf_step(device const float * restrict psi_in   [[buffer(0)]],
                          device       float * restrict psi_out  [[buffer(1)]],
                          device const float * restrict psi_axis [[buffer(2)]],
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
    
    // Boundary cell fast path ensures fixed/zero Dirichlet rules
    if (i == 0 || j == 0 || i == NR - 1 || j == NZ - 1) {
        psi_out[idx] = psi_in[idx];
        return;
    }
    
    // 1. Issue essential base loads early
    float psi_C = psi_in[idx];
    float ax = psi_axis[0];
    
    // 2. Precompute constants optimally while waiting for memory
    float dr = dR;
    float dz = dZ;
    float inv_dr = 1.0f / dr;
    float inv_dz = 1.0f / dz;
    float inv_dr2 = inv_dr * inv_dr;
    float inv_dz2 = inv_dz * inv_dz;
    
    float half_inv_dr = 0.5f * inv_dr;
    float mu0_p_axis_4 = mu0 * p_axis * 4.0f;
    
    float a_C = -2.0f * (inv_dr2 + inv_dz2);
    float a_N = inv_dz2;
    float omega_inv_aC = omega / a_C;
    
    float R = Rmin + float(i) * dr;
    float term2 = half_inv_dr / R;
    
    float a_W = inv_dr2 + term2;
    float a_E = inv_dr2 - term2;
    
    // 3. Compute RHS
    float ax_inv = 1.0f / ax;
    float psi_norm = psi_C * ax_inv;
    
    float rhs = 0.0f;
    if (psi_norm > 0.0f && psi_norm < 1.0f) {
        float R_sq = R * R;
        rhs = -mu0_p_axis_4 * R_sq * psi_norm * (1.0f - psi_norm);
    }
    
    // 4. Defer neighbor loads until variables are exhausted, saving register pressure
    float psi_W = psi_in[idx - 1];
    float psi_E = psi_in[idx + 1];
    float psi_N = psi_in[idx + NR];
    float psi_S = psi_in[idx - NR];
    
    // 5. Integrate using fused multiply-adds mapping correctly to hardware
    float delta_psi = a_C * psi_C;
    delta_psi = fma(a_N, psi_N + psi_S, delta_psi);
    delta_psi = fma(a_E, psi_E, delta_psi);
    delta_psi = fma(a_W, psi_W, delta_psi);
    
    psi_out[idx] = psi_C + omega_inv_aC * (rhs - delta_psi);
}