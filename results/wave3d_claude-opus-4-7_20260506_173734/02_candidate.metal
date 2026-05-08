#include <metal_stdlib>
using namespace metal;

#define BX 32
#define BY 8
#define HALO 1
#define TX (BX + 2*HALO)
#define TY (BY + 2*HALO)

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
    threadgroup float tile[TY][TX];

    uint i = gid.x;
    uint j = gid.y;
    uint k = gid.z;

    uint stride_y = NX;
    uint stride_z = NX * NY;

    // Tile origin in global coords (top-left of interior region)
    uint i0 = tgid.x * BX;
    uint j0 = tgid.y * BY;

    // Linear thread id within threadgroup
    uint tid = lid.y * BX + lid.x;
    uint nthreads = BX * BY;

    // Cooperative load of TX*TY tile (with 1-cell halo) from u_curr.
    // Clamp to domain edges (boundary cells will be overwritten by
    // Dirichlet copy anyway, so clamping is safe for interior threads).
    if (k < NZ) {
        for (uint t = tid; t < TX * TY; t += nthreads) {
            uint tx = t % TX;
            uint ty = t / TX;
            int gi = int(i0) + int(tx) - int(HALO);
            int gj = int(j0) + int(ty) - int(HALO);
            gi = clamp(gi, 0, int(NX) - 1);
            gj = clamp(gj, 0, int(NY) - 1);
            tile[ty][tx] = u_curr[uint(k) * stride_z
                                  + uint(gj) * stride_y
                                  + uint(gi)];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (i >= NX || j >= NY || k >= NZ) return;

    uint idx = k * stride_z + j * stride_y + i;

    if (i == 0u || j == 0u || k == 0u
        || i == NX - 1u || j == NY - 1u || k == NZ - 1u) {
        u_next[idx] = u_curr[idx];
        return;
    }

    uint lx = lid.x + HALO;
    uint ly = lid.y + HALO;

    float c  = tile[ly][lx];
    float xm = tile[ly][lx - 1u];
    float xp = tile[ly][lx + 1u];
    float ym = tile[ly - 1u][lx];
    float yp = tile[ly + 1u][lx];
    float zm = u_curr[idx - stride_z];
    float zp = u_curr[idx + stride_z];

    float lap = xm + xp + ym + yp + zm + zp - 6.0f * c;
    u_next[idx] = 2.0f * c - u_prev[idx] + alpha * lap;
}