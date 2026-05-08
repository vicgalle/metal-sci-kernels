#include <metal_stdlib>
using namespace metal;

kernel void wave_step(device const float *u_prev [[buffer(0)]],
                      device const float *u_curr [[buffer(1)]],
                      device       float *u_next [[buffer(2)]],
                      constant uint      &NX     [[buffer(3)]],
                      constant uint      &NY     [[buffer(4)]],
                      constant uint      &NZ     [[buffer(5)]],
                      constant float     &alpha  [[buffer(6)]],
                      uint3 gid [[thread_position_in_grid]]) {
    uint nx = NX;
    uint ny = NY;
    uint nz = NZ;
    
    uint i = gid.x;
    uint j = gid.y;
    uint k = gid.z;

    if (i >= nx || j >= ny || k >= nz) return;

    uint stride_y = nx;
    uint stride_z = nx * ny;
    uint idx = k * stride_z + j * stride_y + i;

    // Load center cell first; this seamlessly handles the boundary condition
    float c = u_curr[idx];

    // Evaluate boundary condition using bitwise ORs to avoid multiple 
    // short-circuit branches and warp divergence.
    bool is_boundary = (i == 0u) | (j == 0u) | (k == 0u) | 
                       (i == nx - 1u) | (j == ny - 1u) | (k == nz - 1u);

    if (is_boundary) {
        u_next[idx] = c;
        return;
    }

    // Load neighbors (coalesced naturally via L1 cache)
    float xm = u_curr[idx - 1u];
    float xp = u_curr[idx + 1u];
    float ym = u_curr[idx - stride_y];
    float yp = u_curr[idx + stride_y];
    float zm = u_curr[idx - stride_z];
    float zp = u_curr[idx + stride_z];

    float p = u_prev[idx];
    float a = alpha;

    // Rearrange math algebraically to save 1 FMA instruction per cell.
    // Original: lap = sum - 6*c; next = 2*c - p + a * lap
    // Optimized: next = a * sum + (2 - 6*a)*c - p
    float c_coeff = 2.0f - 6.0f * a;
    
    // Grouped additions to maximize ILP
    float sum = (xm + xp) + (ym + yp) + (zm + zp);
    
    u_next[idx] = fma(a, sum, fma(c, c_coeff, -p));
}