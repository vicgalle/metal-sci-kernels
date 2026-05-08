#include <metal_stdlib>
using namespace metal;

kernel void heat_step(device const float * restrict u_in  [[buffer(0)]],
                      device       float * restrict u_out [[buffer(1)]],
                      constant uint      &NX    [[buffer(2)]],
                      constant uint      &NY    [[buffer(3)]],
                      constant float     &alpha [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]]) {
    uint i = gid.x;
    uint j = gid.y;
    
    if (i >= NX || j >= NY) return;

    // Load center value unconditionally for all valid threads to ensure full
    // SIMD participation during the shuffle operations.
    uint idx = j * NX + i;
    float c = u_in[idx];
    
    // Share center values horizontally across the SIMD group.
    float l_shuffle = simd_shuffle_up(c, 1);
    float r_shuffle = simd_shuffle_down(c, 1);
    
    // Share row indices to safely detect when SIMD lanes cross row boundaries.
    uint y_l = simd_shuffle_up(j, 1);
    uint y_r = simd_shuffle_down(j, 1);
    
    // Dirichlet boundary conditions: output unchanged center and exit.
    if (i == 0 || j == 0 || i == NX - 1 || j == NY - 1) {
        u_out[idx] = c;
        return;
    }
    
    // Determine left neighbor: use shuffle if the left lane is within the same row,
    // otherwise fallback to a memory load.
    float l;
    if (lane > 0 && y_l == j) {
        l = l_shuffle;
    } else {
        l = u_in[idx - 1];
    }
    
    // Determine right neighbor: use shuffle if the right lane is within the same row,
    // otherwise fallback to a memory load.
    float r;
    if (lane < 31 && y_r == j) {
        r = r_shuffle;
    } else {
        r = u_in[idx + 1];
    }

    // Vertical neighbors must still be loaded from the L1 cache.
    float d = u_in[idx - NX];
    float u = u_in[idx + NX];

    u_out[idx] = c + alpha * (l + r + d + u - 4.0f * c);
}