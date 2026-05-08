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
    
    // Precompute jump math to bypass integer division in the loop
    uint jump = tgsize;
    uint jump_r = jump / interior_w;
    uint jump_c = jump % interior_w;
    
    uint r = idx / interior_w;
    uint c = idx % interior_w;
    
    while (idx < total_interior) {
        local_max = max(local_max, psi[(r + 1) * NR + (c + 1)]);
        
        idx += jump;
        c += jump_c;
        r += jump_r;
        
        // Fast coordinate modulo bypass
        if (c >= interior_w) {
            c -= interior_w;
            r++;
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
    
    // Boundary cell fast path (preserves Dirichlet zero or fixed boundary edges)
    if (i == 0 || j == 0 || i == NR - 1 || j == NZ - 1) {
        psi_out[idx] = psi_in[idx];
        return;
    }
    
    // 1. Issue memory loads immediately to maximize overlap and allow L1 broadcast
    float psi_C = psi_in[idx];
    float ax    = psi_axis[0];
    float psi_W = psi_in[idx - 1];
    float psi_E = psi_in[idx + 1];
    float psi_N = psi_in[idx + NR];
    float psi_S = psi_in[idx - NR];
    
    // 2. Compute grid constants (perfectly hides memory latency)
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
    
    // 3. Compute position-dependent geometry variables
    float R = Rmin + float(i) * dr;
    float term2 = half_inv_dr / R;
    
    // 4. Evaluate source term (J)
    float ax_inv = 1.0f / ax;
    float psi_norm = psi_C * ax_inv;
    
    float rhs = 0.0f;
    if (psi_norm > 0.0f && psi_norm < 1.0f) {
        float R2 = R * R;
        rhs = -mu0_p_axis_4 * R2 * psi_norm * (1.0f - psi_norm);
    }
    
    // 5. Compute stencil integration with optimized algebraic grouping
    // Equivalent to a_W*psi_W + a_E*psi_E but avoids separate coeff computation
    float sum_WE = psi_W + psi_E;
    float diff_WE = psi_W - psi_E;
    float sum_NS = psi_N + psi_S;
    
    float delta_psi = inv_dr2 * sum_WE + term2 * diff_WE + inv_dz2 * sum_NS + a_C * psi_C;
    float r = rhs - delta_psi;
    
    // 6. Update specific cell for the iteration step
    psi_out[idx] = psi_C + omega_inv_aC * r;
}