#include <metal_stdlib>
using namespace metal;

#define TX 32
#define TY 8

kernel void wave_step(device const float *u_prev [[buffer(0)]],
                      device const float *u_curr [[buffer(1)]],
                      device       float *u_next [[buffer(2)]],
                      constant uint      &NX     [[buffer(3)]],
                      constant uint      &NY     [[buffer(4)]],
                      constant uint      &NZ     [[buffer(5)]],
                      constant float     &alpha  [[buffer(6)]],
                      uint3 gid [[thread_position_in_grid]],
                      uint3 lid [[thread_position_in_threadgroup]]) {
    const uint i = gid.x;
    const uint j = gid.y;
    const uint k = gid.z;

    if (i >= NX || j >= NY || k >= NZ) return;

    const uint stride_y = NX;
    const uint stride_z = NX * NY;
    const uint idx = (k * NY + j) * NX + i;

    // Tile in threadgroup memory: (TY+2) x (TX+2) for XY halo
    threadgroup float tile[TY + 2][TX + 2];

    const uint lx = lid.x; // 0..TX-1
    const uint ly = lid.y; // 0..TY-1

    // Each thread loads its own center cell into tile[ly+1][lx+1]
    float c_val = u_curr[idx];
    tile[ly + 1][lx + 1] = c_val;

    // Load halos cooperatively. Edge threads load extra cells.
    if (lx == 0u) {
        // left halo
        uint ii = (i == 0u) ? 0u : (i - 1u);
        tile[ly + 1][0] = u_curr[(k * NY + j) * NX + ii];
    }
    if (lx == TX - 1u || i == NX - 1u) {
        uint ii = (i + 1u >= NX) ? (NX - 1u) : (i + 1u);
        tile[ly + 1][lx + 2] = u_curr[(k * NY + j) * NX + ii];
    }
    if (ly == 0u) {
        uint jj = (j == 0u) ? 0u : (j - 1u);
        tile[0][lx + 1] = u_curr[(k * NY + jj) * NX + i];
    }
    if (ly == TY - 1u || j == NY - 1u) {
        uint jj = (j + 1u >= NY) ? (NY - 1u) : (j + 1u);
        tile[ly + 2][lx + 1] = u_curr[(k * NY + jj) * NX + i];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    bool is_boundary = (i == 0u) || (j == 0u) || (k == 0u)
                    || (i == NX - 1u) || (j == NY - 1u) || (k == NZ - 1u);

    if (is_boundary) {
        u_next[idx] = c_val;
        return;
    }

    float xm = tile[ly + 1][lx];
    float xp = tile[ly + 1][lx + 2];
    float ym = tile[ly][lx + 1];
    float yp = tile[ly + 2][lx + 1];
    float zm = u_curr[idx - stride_z];
    float zp = u_curr[idx + stride_z];

    float lap = xm + xp + ym + yp + zm + zp - 6.0f * c_val;
    u_next[idx] = fma(alpha, lap, 2.0f * c_val - u_prev[idx]);
}