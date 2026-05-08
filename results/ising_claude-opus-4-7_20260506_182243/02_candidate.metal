#include <metal_stdlib>
using namespace metal;

inline uint mix32(uint x) {
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    x = x ^ (x >> 16);
    return x;
}

#define TX 32
#define TY 8

kernel void ising_step(device       char  *spins    [[buffer(0)]],
                       device const float *p_accept [[buffer(1)]],
                       constant uint  &NX           [[buffer(2)]],
                       constant uint  &NY           [[buffer(3)]],
                       constant uint  &step_idx     [[buffer(4)]],
                       constant uint  &seed         [[buffer(5)]],
                       uint2 gid  [[thread_position_in_grid]],
                       uint2 tgid [[threadgroup_position_in_grid]],
                       uint2 lid  [[thread_position_in_threadgroup]]) {
    // Tile with halo of 1 on each side: (TX+2) x (TY+2)
    threadgroup char tile[(TY + 2)][(TX + 2)];
    threadgroup float pa_tab[5];

    uint lx = lid.x;
    uint ly = lid.y;
    uint flat = ly * TX + lx;

    if (flat < 5u) {
        pa_tab[flat] = p_accept[flat];
    }

    // Tile origin in global coords (top-left of interior)
    uint base_x = tgid.x * TX;
    uint base_y = tgid.y * TY;

    uint i = base_x + lx;
    uint j = base_y + ly;

    // Cooperative load: each thread loads tile[ly+1][lx+1] (its own site).
    // Edge threads also load halo cells.
    auto load_site = [&](int gx, int gy) -> char {
        uint gxw = (uint)((gx + (int)NX) % (int)NX);
        uint gyw = (uint)((gy + (int)NY) % (int)NY);
        return spins[gyw * NX + gxw];
    };

    // Load interior cell
    if (i < NX && j < NY) {
        tile[ly + 1][lx + 1] = load_site((int)i, (int)j);
    } else {
        tile[ly + 1][lx + 1] = 0;
    }

    // Halo: left/right columns
    if (lx == 0) {
        tile[ly + 1][0] = load_site((int)base_x - 1, (int)j);
    }
    if (lx == TX - 1) {
        tile[ly + 1][TX + 1] = load_site((int)(base_x + TX), (int)j);
    }
    // Halo: top/bottom rows
    if (ly == 0) {
        tile[0][lx + 1] = load_site((int)i, (int)base_y - 1);
    }
    if (ly == TY - 1) {
        tile[TY + 1][lx + 1] = load_site((int)i, (int)(base_y + TY));
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (i >= NX || j >= NY) return;

    uint color = step_idx & 1u;
    if (((i + j) & 1u) != color) return;

    int s  = (int)tile[ly + 1][lx + 1];
    int sl = (int)tile[ly + 1][lx    ];
    int sr = (int)tile[ly + 1][lx + 2];
    int sd = (int)tile[ly    ][lx + 1];
    int su = (int)tile[ly + 2][lx + 1];

    int h    = sl + sr + sd + su;
    int prod = s * h;
    int idx  = (prod + 4) >> 1;
    float pa = pa_tab[idx];

    uint site_idx = j * NX + i;
    uint x = seed + step_idx * 0x9E3779B9u;
    x = mix32(x);
    x = mix32(x ^ site_idx);
    float u = float(x >> 8) * (1.0f / 16777216.0f);

    if (u < pa) {
        spins[site_idx] = (char)(-s);
    }
}