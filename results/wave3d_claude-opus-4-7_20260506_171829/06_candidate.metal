#include <metal_stdlib>
using namespace metal;

#define TX 32
#define TY 8
#define HALO 1
#define SX (TX + 2*HALO)
#define SY (TY + 2*HALO)

kernel void wave_step(device const float *u_prev [[buffer(0)]],
                      device const float *u_curr [[buffer(1)]],
                      device       float *u_next [[buffer(2)]],
                      constant uint      &NX     [[buffer(3)]],
                      constant uint      &NY     [[buffer(4)]],
                      constant uint      &NZ     [[buffer(5)]],
                      constant float     &alpha  [[buffer(6)]],
                      uint3 gid  [[thread_position_in_grid]],
                      uint3 tid  [[thread_position_in_threadgroup]],
                      uint3 tgid [[threadgroup_position_in_grid]]) {
    threadgroup float tile[SY][SX];

    const uint i = gid.x;
    const uint j = gid.y;
    const uint k = gid.z;

    const uint lx = tid.x;       // 0..TX-1
    const uint ly = tid.y;       // 0..TY-1
    const uint sx = lx + HALO;
    const uint sy = ly + HALO;

    const uint stride_y = NX;
    const uint stride_z = NX * NY;

    const uint NXm1 = NX - 1u;
    const uint NYm1 = NY - 1u;
    const uint NZm1 = NZ - 1u;

    const bool in_bounds = (i < NX) && (j < NY) && (k < NZ);
    const uint lin = in_bounds ? ((k * NY + j) * NX + i) : 0u;

    // Cooperative load of the (TX+2) x (TY+2) tile for this Z-slice.
    // Center.
    if (in_bounds) {
        tile[sy][sx] = u_curr[lin];
    } else {
        tile[sy][sx] = 0.0f;
    }

    // X halos (left/right columns).
    if (lx == 0u) {
        uint ii = (i == 0u) ? 0u : (i - 1u);
        uint jj = (j < NY) ? j : (NYm1);
        uint kk = (k < NZ) ? k : (NZm1);
        tile[sy][0] = u_curr[(kk * NY + jj) * NX + ii];
    }
    if (lx == TX - 1u) {
        uint ii = (i + 1u < NX) ? (i + 1u) : NXm1;
        uint jj = (j < NY) ? j : NYm1;
        uint kk = (k < NZ) ? k : NZm1;
        tile[sy][SX - 1u] = u_curr[(kk * NY + jj) * NX + ii];
    }

    // Y halos (top/bottom rows).
    if (ly == 0u) {
        uint jj = (j == 0u) ? 0u : (j - 1u);
        uint ii = (i < NX) ? i : NXm1;
        uint kk = (k < NZ) ? k : NZm1;
        tile[0][sx] = u_curr[(kk * NY + jj) * NX + ii];
    }
    if (ly == TY - 1u) {
        uint jj = (j + 1u < NY) ? (j + 1u) : NYm1;
        uint ii = (i < NX) ? i : NXm1;
        uint kk = (k < NZ) ? k : NZm1;
        tile[SY - 1u][sx] = u_curr[(kk * NY + jj) * NX + ii];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!in_bounds) return;

    // Boundary: Dirichlet — copy u_curr -> u_next.
    if (i == 0u || j == 0u || k == 0u || i == NXm1 || j == NYm1 || k == NZm1) {
        u_next[lin] = tile[sy][sx];
        return;
    }

    float c  = tile[sy][sx];
    float xm = tile[sy][sx - 1];
    float xp = tile[sy][sx + 1];
    float ym = tile[sy - 1][sx];
    float yp = tile[sy + 1][sx];
    float zm = u_curr[lin - stride_z];
    float zp = u_curr[lin + stride_z];

    float lap = (xm + xp) + (ym + yp) + (zm + zp) - 6.0f * c;
    float p   = u_prev[lin];
    u_next[lin] = fma(alpha, lap, fma(2.0f, c, -p));
}