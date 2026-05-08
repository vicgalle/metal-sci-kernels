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
                      uint tid [[thread_index_in_simdgroup]])
{
    uint nx = NX;
    uint ny = NY;
    uint nz = NZ;

    uint i = gid.x;
    uint j = gid.y;
    uint k = gid.z;

    // Only true out-of-bounds threads exit early, keeping the warp 
    // mostly uniform for safe SIMD shuffles below.
    if (i >= nx || j >= ny || k >= nz) {
        return;
    }

    uint stride_y = nx;
    uint stride_z = nx * ny;
    uint idx = k * stride_z + j * stride_y + i;

    float c = u_curr[idx];

    // Uniformly shuffle X neighbors across all active threads.
    uint i_left = simd_shuffle_up(i, 1);
    float xm = simd_shuffle_up(c, 1);
    
    uint i_right = simd_shuffle_down(i, 1);
    float xp = simd_shuffle_down(c, 1);

    bool is_boundary = (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1);

    if (is_boundary) {
        // Dirichlet boundary: copy and exit.
        // Computed shuffle values are safely discarded.
        u_next[idx] = c;
    } else {
        // Resolve missing edges from SIMD boundaries or grid row wrap-arounds.
        // Because we are in the `else` branch, i > 0 and i < nx - 1 are guaranteed safe.
        if (tid == 0 || i_left != i - 1) {
            xm = u_curr[idx - 1];
        }
        if (tid == 31 || i_right != i + 1) {
            xp = u_curr[idx + 1];
        }

        // Deferring these loads to only internal cells saves bandwidth on boundaries.
        float ym = u_curr[idx - stride_y];
        float yp = u_curr[idx + stride_y];
        float zm = u_curr[idx - stride_z];
        float zp = u_curr[idx + stride_z];

        float p = u_prev[idx];

        float lap_sum = (xm + xp) + (ym + yp) + (zm + zp);
        float a = alpha;
        float c_coeff = 2.0f - 6.0f * a;
        
        // Algebraically equivalent to: 2*c - p + a * (lap_sum - 6*c)
        u_next[idx] = fma(a, lap_sum, fma(c_coeff, c, -p));
    }
}