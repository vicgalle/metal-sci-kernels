#include <metal_stdlib>
using namespace metal;

#define BX 32
#define BY 8
#define BZ 4

kernel void wave_step(device const float *u_prev [[buffer(0)]],
                      device const float *u_curr [[buffer(1)]],
                      device       float *u_next [[buffer(2)]],
                      constant uint      &NX     [[buffer(3)]],
                      constant uint      &NY     [[buffer(4)]],
                      constant uint      &NZ     [[buffer(5)]],
                      constant float     &alpha  [[buffer(6)]],
                      uint3 gid [[thread_position_in_grid]],
                      uint3 lid [[thread_position_in_threadgroup]]) {
    threadgroup float tile[BZ][BY+2][BX+2];

    uint i = gid.x;
    uint j = gid.y;
    uint k = gid.z;

    uint stride_y = NX;
    uint stride_z = NX * NY;

    uint lx = lid.x + 1u;
    uint ly = lid.y + 1u;
    uint lz = lid.z;

    bool in_bounds = (i < NX) && (j < NY) && (k < NZ);

    // Center load
    if (in_bounds) {
        tile[lz][ly][lx] = u_curr[k * stride_z + j * stride_y + i];
    } else {
        tile[lz][ly][lx] = 0.0f;
    }

    // X halos
    if (lid.x == 0u) {
        uint ii = (i == 0u) ? 0u : i - 1u;
        if (j < NY && k < NZ) {
            tile[lz][ly][0] = u_curr[k * stride_z + j * stride_y + ii];
        } else {
            tile[lz][ly][0] = 0.0f;
        }
    }
    if (lid.x == BX - 1u || i == NX - 1u) {
        uint ii = (i + 1u >= NX) ? NX - 1u : i + 1u;
        if (j < NY && k < NZ) {
            tile[lz][ly][lx + 1u] = u_curr[k * stride_z + j * stride_y + ii];
        } else {
            tile[lz][ly][lx + 1u] = 0.0f;
        }
    }
    // Y halos
    if (lid.y == 0u) {
        uint jj = (j == 0u) ? 0u : j - 1u;
        if (i < NX && k < NZ) {
            tile[lz][0][lx] = u_curr[k * stride_z + jj * stride_y + i];
        } else {
            tile[lz][0][lx] = 0.0f;
        }
    }
    if (lid.y == BY - 1u || j == NY - 1u) {
        uint jj = (j + 1u >= NY) ? NY - 1u : j + 1u;
        if (i < NX && k < NZ) {
            tile[lz][ly + 1u][lx] = u_curr[k * stride_z + jj * stride_y + i];
        } else {
            tile[lz][ly + 1u][lx] = 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!in_bounds) return;

    uint idx = k * stride_z + j * stride_y + i;

    if (i == 0u || j == 0u || k == 0u
        || i == NX - 1u || j == NY - 1u || k == NZ - 1u) {
        u_next[idx] = u_curr[idx];
        return;
    }

    float c  = tile[lz][ly][lx];
    float xm = tile[lz][ly][lx - 1u];
    float xp = tile[lz][ly][lx + 1u];
    float ym = tile[lz][ly - 1u][lx];
    float yp = tile[lz][ly + 1u][lx];
    float zm = u_curr[idx - stride_z];
    float zp = u_curr[idx + stride_z];

    float lap = xm + xp + ym + yp + zm + zp - 6.0f * c;
    u_next[idx] = 2.0f * c - u_prev[idx] + alpha * lap;
}