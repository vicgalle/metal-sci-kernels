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

    float c = u_curr[idx];

    // Branchless boundary evaluation using bitwise OR to avoid warp divergence
    bool is_boundary = (i == 0u) | (i == nx - 1u) | 
                       (j == 0u) | (j == ny - 1u) | 
                       (k == 0u) | (k == nz - 1u);
    if (is_boundary) {
        u_next[idx] = c;
        return;
    }

    // Explicit parallel loads to maximize memory-level parallelism
    float xm = u_curr[idx - 1u];
    float xp = u_curr[idx + 1u];
    float ym = u_curr[idx - stride_y];
    float yp = u_curr[idx + stride_y];
    float zm = u_curr[idx - stride_z];
    float zp = u_curr[idx + stride_z];
    float p  = u_prev[idx];

    // Balanced addition tree
    float sum_x = xm + xp;
    float sum_y = ym + yp;
    float sum_z = zm + zp;
    
    // Explicit Laplacian preserves spatial derivative precision
    float lap = (sum_x + sum_y + sum_z) - 6.0f * c;
    
    float a = alpha;
    
    // Contract final integration into two FMA instructions
    u_next[idx] = fma(a, lap, fma(2.0f, c, -p));
}