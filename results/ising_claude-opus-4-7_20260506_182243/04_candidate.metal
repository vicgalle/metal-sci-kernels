#include <metal_stdlib>
using namespace metal;

inline uint mix32(uint x) {
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    x = x ^ (x >> 16);
    return x;
}

kernel void ising_step(device       char  *spins    [[buffer(0)]],
                       device const float *p_accept [[buffer(1)]],
                       constant uint  &NX           [[buffer(2)]],
                       constant uint  &NY           [[buffer(3)]],
                       constant uint  &step_idx     [[buffer(4)]],
                       constant uint  &seed         [[buffer(5)]],
                       uint2 gid [[thread_position_in_grid]],
                       uint   lid [[thread_index_in_threadgroup]]) {
    threadgroup float pa_tab[5];
    if (lid < 5u) {
        pa_tab[lid] = p_accept[lid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint i = gid.x;
    uint j = gid.y;
    if (i >= NX || j >= NY) return;

    uint color = step_idx & 1u;
    if (((i + j) & 1u) != color) return;

    // Branchless periodic neighbours.
    uint im = (i == 0u)        ? (NX - 1u) : (i - 1u);
    uint ip = (i == NX - 1u)   ? 0u        : (i + 1u);
    uint jm = (j == 0u)        ? (NY - 1u) : (j - 1u);
    uint jp = (j == NY - 1u)   ? 0u        : (j + 1u);

    uint row   = j  * NX;
    uint row_m = jm * NX;
    uint row_p = jp * NX;

    int s  = (int)spins[row   + i ];
    int sl = (int)spins[row   + im];
    int sr = (int)spins[row   + ip];
    int sd = (int)spins[row_m + i ];
    int su = (int)spins[row_p + i ];

    int h    = sl + sr + sd + su;
    int prod = s * h;
    int idx  = (prod + 4) >> 1;
    float pa = pa_tab[idx];

    uint site_idx = row + i;
    uint x = seed + step_idx * 0x9E3779B9u;
    x = mix32(x);
    x = mix32(x ^ site_idx);
    float u = float(x >> 8) * (1.0f / 16777216.0f);

    if (u < pa) {
        spins[site_idx] = (char)(-s);
    }
}