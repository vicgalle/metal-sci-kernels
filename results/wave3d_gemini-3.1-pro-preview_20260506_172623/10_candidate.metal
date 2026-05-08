#include <metal_stdlib>
using namespace metal;

kernel void wave_step(device const float *u_prev [[buffer(0)]],
                      device const float *u_curr [[buffer(1)]],
                      device       float *u_next [[buffer(2)]],
                      constant uint      &NX     [[buffer(3)]],
                      constant uint      &NY     [[buffer(4)]],
                      constant uint      &NZ     [[buffer(5)]],
                      constant float     &alpha  [[buffer(6)]],
                      uint3 gid [[thread_position_in_grid]])
{
    uint nx = NX;
    uint ny = NY;
    uint nz = NZ;

    uint i = gid.x;
    uint j = gid.y;
    uint k = gid.z;

    if (i >= nx || j >= ny || k >= nz) {
        return;
    }

    uint stride_y = nx;
    uint stride_z = nx * ny;
    uint idx = k * stride_z + j * stride_y + i;

    // Load center cell once. We use this for both the internal stencil and 
    // the Dirichlet boundary condition to save a redundant global read.
    float c = u_curr[idx];

    // Dirichlet boundary: copy current state to next state and exit.
    if (i == 0 || j == 0 || k == 0 || i == nx - 1 || j == ny - 1 || k == nz - 1) {
        u_next[idx] = c;
        return;
    }

    // Load spatial neighbors (hardware L1 cache perfectly absorbs this)
    float xm = u_curr[idx - 1];
    float xp = u_curr[idx + 1];
    float ym = u_curr[idx - stride_y];
    float yp = u_curr[idx + stride_y];
    float zm = u_curr[idx - stride_z];
    float zp = u_curr[idx + stride_z];

    float p = u_prev[idx];

    // Adder tree to maximize instruction-level parallelism
    float sum_x = xm + xp;
    float sum_y = ym + yp;
    float sum_z = zm + zp;
    float lap   = sum_x + sum_y + sum_z;

    // Fold the -6*c into the center coefficient to save ALU instructions
    float a = alpha;
    float c_coeff = 2.0f - 6.0f * a;
    
    // Mathematically equivalent to: u_next = a * lap + c_coeff * c - p
    u_next[idx] = fma(a, lap, fma(c_coeff, c, -p));
}