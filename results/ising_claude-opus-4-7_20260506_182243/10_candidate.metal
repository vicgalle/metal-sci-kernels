#include <metal_stdlib>
using namespace metal;

inline uint mix32(uint x) {
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    x = x ^ (x >> 16);
    return x;
}

#define TX 32u
#define TY 8u
#define TW (TX + 2u)
#define TH (TY + 2u)

kernel void ising_step(
    device       char  *spins    [[buffer(0)]],
    device const float *p_accept [[buffer(1)]],
    constant uint  &NX           [[buffer(2)]],
    constant uint  &NY           [[buffer(3)]],
    constant uint  &step_idx     [[buffer(4)]],
    constant uint  &seed         [[buffer(5)]],
    uint2 gid  [[thread_position_in_grid]],
    uint2 lid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    threadgroup float pa_tab[5];
    threadgroup char  tile[TH * TW];

    uint lx = lid.x;
    uint ly = lid.y;
    uint flat = ly * TX + lx;
    const uint total = TX * TY; // 256

    if (flat < 5u) {
        pa_tab[flat] = p_accept[flat];
    }

    uint i0 = tgid.x * TX;
    uint j0 = tgid.y * TY;

    // Cooperative halo load: TH * TW = 10 * 34 = 340 cells, 256 threads.
    const uint tile_n = TH * TW;
    for (uint k = flat; k < tile_n; k += total) {
        uint ty_ = k / TW;
        uint tx_ = k - ty_ * TW;

        int gi_s = (int)i0 + (int)tx_ - 1;
        int gj_s = (int)j0 + (int)ty_ - 1;
        if (gi_s < 0)        gi_s += (int)NX;
        if (gi_s >= (int)NX) gi_s -= (int)NX;
        if (gj_s < 0)        gj_s += (int)NY;
        if (gj_s >= (int)NY) gj_s -= (int)NY;

        tile[ty_ * TW + tx_] = spins[(uint)gj_s * NX + (uint)gi_s];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint i = gid.x;
    uint j = gid.y;
    if (i >= NX || j >= NY) return;

    uint color = step_idx & 1u;
    if (((i + j) & 1u) != color) return;

    uint tx = lx + 1u;
    uint ty = ly + 1u;

    int s  = (int)tile[ty * TW + tx];
    int sl = (int)tile[ty * TW + (tx - 1u)];
    int sr = (int)tile[ty * TW + (tx + 1u)];
    int sd = (int)tile[(ty - 1u) * TW + tx];
    int su = (int)tile[(ty + 1u) * TW + tx];

    int h    = sl + sr + sd + su;
    int prod = s * h;
    int idx  = (prod + 4) >> 1;
    float pa = pa_tab[idx];

    uint site_idx = j * NX + i;
    uint x = seed + step_idx * 0x9E3779B9u;
    x = mix32(x);
    x = mix32(x ^ site_idx);
    float u = float(x >> 8) * (1.0f / 16777216.0f);

    char new_s = (char)(-s);
    char old_s = (char)s;
    spins[site_idx] = (u < pa) ? new_s : old_s;
}