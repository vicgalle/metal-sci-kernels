#include <metal_stdlib>
using namespace metal;

#define TX 16
#define TY 16

kernel void wave_step(device const float *u_prev [[buffer(0)]],
                      device const float *u_curr [[buffer(1)]],
                      device       float *u_next [[buffer(2)]],
                      constant uint      &NX     [[buffer(3)]],
                      constant uint      &NY     [[buffer(4)]],
                      constant uint      &NZ     [[buffer(5)]],
                      constant float     &alpha  [[buffer(6)]],
                      uint3 gid  [[thread_position_in_grid]],
                      uint3 lid  [[thread_position_in_threadgroup]],
                      uint3 tgid [[threadgroup_position_in_grid]]) {
    const uint i = gid.x;
    const uint j = gid.y;
    const uint k = gid.z;

    const uint lx = lid.x;
    const uint ly = lid.y;

    // Tile in shared memory with 1-cell halo on each side
    threadgroup float tile[TY + 2][TX + 2];

    const uint stride_y = NX;
    const uint stride_z = NX * NY;

    // Base coordinates of this threadgroup's tile origin (in global coords)
    const uint i0 = tgid.x * TX;
    const uint j0 = tgid.y * TY;

    // Each thread loads its center cell into tile[ly+1][lx+1] (clamped)
    auto load_curr = [&](int gi, int gj) -> float {
        int ci = clamp(gi, 0, int(NX) - 1);
        int cj = clamp(gj, 0, int(NY) - 1);
        return u_curr[k * stride_z + uint(cj) * stride_y + uint(ci)];
    };

    if (k < NZ) {
        // Center
        tile[ly + 1][lx + 1] = load_curr(int(i0 + lx), int(j0 + ly));

        // Halo: left/right columns
        if (lx == 0) {
            tile[ly + 1][0] = load_curr(int(i0) - 1, int(j0 + ly));
        }
        if (lx == TX - 1) {
            tile[ly + 1][TX + 1] = load_curr(int(i0 + TX), int(j0 + ly));
        }
        // Halo: top/bottom rows
        if (ly == 0) {
            tile[0][lx + 1] = load_curr(int(i0 + lx), int(j0) - 1);
        }
        if (ly == TY - 1) {
            tile[TY + 1][lx + 1] = load_curr(int(i0 + lx), int(j0 + TY));
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (i >= NX || j >= NY || k >= NZ) return;

    const uint NXm1 = NX - 1u;
    const uint NYm1 = NY - 1u;
    const uint NZm1 = NZ - 1u;
    const uint idx = k * stride_z + j * stride_y + i;

    if (i == 0u || j == 0u || k == 0u
        || i == NXm1 || j == NYm1 || k == NZm1) {
        u_next[idx] = u_curr[idx];
        return;
    }

    float c  = tile[ly + 1][lx + 1];
    float xm = tile[ly + 1][lx];
    float xp = tile[ly + 1][lx + 2];
    float ym = tile[ly][lx + 1];
    float yp = tile[ly + 2][lx + 1];
    float zm = u_curr[idx - stride_z];
    float zp = u_curr[idx + stride_z];
    float p  = u_prev[idx];

    float sxy = (xm + xp) + (ym + yp);
    float sz  = zm + zp;
    float lap = (sxy + sz) - 6.0f * c;

    u_next[idx] = fma(alpha, lap, fma(2.0f, c, -p));
}