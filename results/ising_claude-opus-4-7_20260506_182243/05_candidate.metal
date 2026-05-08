#include <metal_stdlib>
using namespace metal;

inline uint mix32(uint x) {
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    x = x ^ (x >> 16);
    return x;
}

#define TGW 16
#define TGH 16

kernel void ising_step(device       char  *spins    [[buffer(0)]],
                       device const float *p_accept [[buffer(1)]],
                       constant uint  &NX           [[buffer(2)]],
                       constant uint  &NY           [[buffer(3)]],
                       constant uint  &step_idx     [[buffer(4)]],
                       constant uint  &seed         [[buffer(5)]],
                       uint2 gid  [[thread_position_in_grid]],
                       uint2 lid2 [[thread_position_in_threadgroup]],
                       uint  lid  [[thread_index_in_threadgroup]],
                       uint2 tgid [[threadgroup_position_in_grid]])
{
    threadgroup float pa_tab[5];
    threadgroup char  tile[(TGH + 2) * (TGW + 2)];

    // Cache p_accept (5 entries).
    if (lid < 5u) {
        pa_tab[lid] = p_accept[lid];
    }

    // Origin of the tile in lattice coords (top-left of inner region).
    int base_i = (int)(tgid.x * TGW);
    int base_j = (int)(tgid.y * TGH);
    int nx = (int)NX;
    int ny = (int)NY;

    // Cooperatively load (TGH+2) x (TGW+2) tile including halo.
    // Total = 324 elements, threadgroup has 256 threads => up to 2 per thread.
    const uint TILE_W = TGW + 2;
    const uint TILE_H = TGH + 2;
    const uint TILE_N = TILE_W * TILE_H; // 324
    const uint TG_N   = TGW * TGH;       // 256

    for (uint t = lid; t < TILE_N; t += TG_N) {
        uint ty = t / TILE_W;
        uint tx = t - ty * TILE_W;
        int gx = base_i + (int)tx - 1;
        int gy = base_j + (int)ty - 1;
        // Periodic wrap.
        if (gx < 0)      gx += nx;
        else if (gx >= nx) gx -= nx;
        if (gy < 0)      gy += ny;
        else if (gy >= ny) gy -= ny;
        tile[t] = spins[gy * nx + gx];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint i = gid.x;
    uint j = gid.y;
    if (i >= NX || j >= NY) return;

    uint color = step_idx & 1u;
    bool active = (((i + j) & 1u) == color);

    // Local tile coordinates of this site (offset by +1 for halo).
    uint tx = lid2.x + 1u;
    uint ty = lid2.y + 1u;

    int s  = (int)tile[ty * TILE_W + tx];
    int sl = (int)tile[ty * TILE_W + (tx - 1u)];
    int sr = (int)tile[ty * TILE_W + (tx + 1u)];
    int sd = (int)tile[(ty - 1u) * TILE_W + tx];
    int su = (int)tile[(ty + 1u) * TILE_W + tx];

    int h    = sl + sr + sd + su;
    int prod = s * h;
    int idx  = (prod + 4) >> 1;
    float pa = pa_tab[idx];

    uint site_idx = j * NX + i;
    uint x = seed + step_idx * 0x9E3779B9u;
    x = mix32(x);
    x = mix32(x ^ site_idx);
    float u = float(x >> 8) * (1.0f / 16777216.0f);

    if (active && (u < pa)) {
        spins[site_idx] = (char)(-s);
    }
}