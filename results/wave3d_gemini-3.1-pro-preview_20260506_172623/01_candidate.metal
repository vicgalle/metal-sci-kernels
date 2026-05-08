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
                      uint lane [[thread_index_in_simdgroup]],
                      uint lane_width [[threads_per_simdgroup]]) {
    uint i = gid.x;
    uint j = gid.y;
    uint k = gid.z;
    
    // Bounds check
    if (i >= NX || j >= NY || k >= NZ) return;

    uint idx = (k * NY + j) * NX + i;
    float c  = u_curr[idx];

    // SIMD shuffles for X-axis neighbors MUST be called before divergent boundary returns
    // so that all threads within the domain boundary participate actively.
    float c_up   = simd_shuffle_up(c, 1);
    uint  idx_up = simd_shuffle_up(idx, 1);

    float c_down   = simd_shuffle_down(c, 1);
    uint  idx_down = simd_shuffle_down(idx, 1);

    // Dirichlet boundary conditions: edge cells copy u_curr unchanged
    if (i == 0u || j == 0u || k == 0u ||
        i == NX - 1u || j == NY - 1u || k == NZ - 1u) {
        u_next[idx] = c;
        return;
    }

    uint stride_y = NX;
    uint stride_z = NX * NY;

    // Use shuffled values if they correspond to the correct neighbor (handles threadgroup wrapping)
    float xm = (lane > 0 && idx_up == idx - 1u) ? c_up : u_curr[idx - 1u];
    float xp = (lane < lane_width - 1u && idx_down == idx + 1u) ? c_down : u_curr[idx + 1u];
    
    // Y and Z axis neighbors
    float ym = u_curr[idx - stride_y];
    float yp = u_curr[idx + stride_y];
    float zm = u_curr[idx - stride_z];
    float zp = u_curr[idx + stride_z];

    // Explicit FMAs for maximum ALU throughput
    float sum = xm + xp + ym + yp + zm + zp;
    float lap = fma(-6.0f, c, sum);
    
    u_next[idx] = fma(alpha, lap, fma(2.0f, c, -u_prev[idx]));
}