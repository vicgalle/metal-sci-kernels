#include <metal_stdlib>
using namespace metal;

kernel void wave_step(device const float * restrict u_prev [[buffer(0)]],
                      device const float * restrict u_curr [[buffer(1)]],
                      device       float * restrict u_next [[buffer(2)]],
                      constant uint      &NX     [[buffer(3)]],
                      constant uint      &NY     [[buffer(4)]],
                      constant uint      &NZ     [[buffer(5)]],
                      constant float     &alpha  [[buffer(6)]],
                      uint3 gid [[thread_position_in_grid]],
                      uint3 tpos [[thread_position_in_threadgroup]],
                      uint3 tsize [[threads_per_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint simdW [[thread_execution_width]]) {
    
    uint nx = NX;
    uint ny = NY;
    uint nz = NZ;
    
    uint i = gid.x;
    uint j = gid.y;
    uint k = gid.z;
    
    // Bounds check
    if (i >= nx || j >= ny || k >= nz) return;

    uint idx = (k * ny + j) * nx + i;
    
    // Every active thread loads its center element
    float c = u_curr[idx];
    
    uint tx = tpos.x;
    
    // SIMD group cooperation for X-axis neighbors (bypasses memory loads)
    float xm = simd_shuffle_up(c, 1);
    // Overwrite with memory load if this thread is the first in its row or SIMD group
    if ((tx == 0 || lane == 0) && i > 0) {
        xm = u_curr[idx - 1];
    }
    
    float xp = simd_shuffle_down(c, 1);
    // Overwrite with memory load if this thread is the last in its row or SIMD group
    if ((tx == tsize.x - 1 || lane == simdW - 1) && i < nx - 1) {
        xp = u_curr[idx + 1];
    }
    
    // Dirichlet boundary condition
    // We evaluate this AFTER the shuffle so boundary cells can supply 'c' to interior cells
    bool is_boundary = (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1);
    if (is_boundary) {
        u_next[idx] = c;
        return;
    }
    
    // Only interior cells proceed to load Y and Z neighbors and compute the update
    uint stride_y = nx;
    uint stride_z = nx * ny;
    
    float p = u_prev[idx];
    
    float ym = u_curr[idx - stride_y];
    float yp = u_curr[idx + stride_y];
    float zm = u_curr[idx - stride_z];
    float zp = u_curr[idx + stride_z];
    
    float lap = (xm + xp) + (ym + yp) + (zm + zp) - 6.0f * c;
    float a = alpha;
    
    u_next[idx] = fma(a, lap, fma(2.0f, c, -p));
}