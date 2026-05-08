#include <metal_stdlib>
using namespace metal;

kernel void wave_step(device const float *u_prev [[buffer(0)]],
                      device const float *u_curr [[buffer(1)]],
                      device       float *u_next [[buffer(2)]],
                      constant uint      &NX     [[buffer(3)]],
                      constant uint      &NY     [[buffer(4)]],
                      constant uint      &NZ     [[buffer(5)]],
                      constant float     &alpha  [[buffer(6)]],
                      uint3 gid [[thread_position_in_grid]],
                      uint3 lid [[thread_position_in_threadgroup]],
                      uint3 lsize [[threads_per_threadgroup]]) {
    uint i = gid.x;
    uint j = gid.y;
    uint k = gid.z;

    if (i >= NX || j >= NY || k >= NZ) return;

    uint idx = (k * NY + j) * NX + i;

    // Evaluate boundary condition branchlessly using bitwise OR
    bool is_boundary = (i == 0u) | (j == 0u) | (k == 0u) |
                       (i == NX - 1u) | (j == NY - 1u) | (k == NZ - 1u);

    float c = u_curr[idx];

    // Use warp-level intrinsics to share X-axis neighbors.
    // This eliminates unaligned L1 cache accesses for most threads.
    // Threads at the boundary of the threadgroup fall back to a direct memory load.
    float xm = simd_shuffle_up(c, 1);
    if (lid.x == 0u && i > 0u) xm = u_curr[idx - 1u];

    float xp = simd_shuffle_down(c, 1);
    if (lid.x == lsize.x - 1u && i < NX - 1u) xp = u_curr[idx + 1u];

    if (is_boundary) {
        // Dirichlet boundary: copy current state
        u_next[idx] = c;
    } else {
        uint stride_y = NX;
        uint stride_z = NX * NY;

        // Since i varies fastest, these Y/Z offsets are perfectly aligned for the warp
        float ym = u_curr[idx - stride_y];
        float yp = u_curr[idx + stride_y];
        float zm = u_curr[idx - stride_z];
        float zp = u_curr[idx + stride_z];
        float p = u_prev[idx];

        // Adder tree to maximize instruction-level parallelism
        float sum = (xm + xp) + (ym + yp) + (zm + zp);
        
        // Fold the -6*c into the center coefficient to save 1 FMA instruction
        float c_coeff = 2.0f - 6.0f * alpha;
        
        u_next[idx] = fma(alpha, sum, fma(c_coeff, c, -p));
    }
}