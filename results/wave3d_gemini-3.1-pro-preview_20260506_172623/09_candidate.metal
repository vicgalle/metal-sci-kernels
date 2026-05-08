#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(256)]]
kernel void wave_step(device const float * restrict u_prev [[buffer(0)]],
                      device const float * restrict u_curr [[buffer(1)]],
                      device       float * restrict u_next [[buffer(2)]],
                      constant uint      &NX     [[buffer(3)]],
                      constant uint      &NY     [[buffer(4)]],
                      constant uint      &NZ     [[buffer(5)]],
                      constant float     &alpha  [[buffer(6)]],
                      uint3 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint j = gid.y;
    uint k = gid.z;

    if (i >= NX || j >= NY || k >= NZ) return;

    uint idx = (k * NY + j) * NX + i;

    // Dirichlet boundary: copy current state to next state and exit.
    // Early exit perfectly avoids bounds-checking overhead for neighbors
    // and saves significant memory bandwidth and ALU cycles.
    if (i == 0u || j == 0u || k == 0u || i == NX - 1u || j == NY - 1u || k == NZ - 1u) {
        u_next[idx] = u_curr[idx];
        return;
    }

    uint stride_y = NX;
    uint stride_z = NX * NY;

    // Load center and time-previous
    float c = u_curr[idx];
    float p = u_prev[idx];

    // Load spatial neighbors. Apple Silicon L1 cache handles the spatial
    // locality of X and Y neighbors extremely efficiently.
    float xm = u_curr[idx - 1u];
    float xp = u_curr[idx + 1u];
    float ym = u_curr[idx - stride_y];
    float yp = u_curr[idx + stride_y];
    float zm = u_curr[idx - stride_z];
    float zp = u_curr[idx + stride_z];

    // Adder tree to maximize instruction-level parallelism and reduce latency
    float sum_x = xm + xp;
    float sum_y = ym + yp;
    float sum_z = zm + zp;
    float lap = (sum_x + sum_y) + sum_z;

    // Fold the -6*c into the center coefficient to save FMA instructions.
    // mathematically: u_next = alpha * lap + (2 - 6*alpha) * c - p
    float a = alpha;
    float c_coeff = 2.0f - 6.0f * a;
    
    u_next[idx] = fma(a, lap, fma(c_coeff, c, -p));
}