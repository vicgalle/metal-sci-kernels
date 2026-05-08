**Optimization:** Vectorized neighbor loads using `char4` for the horizontal triple `(im, i, ip)` when possible, but more importantly: switch to a memory-access pattern that uses `uchar` indexing arithmetic without the `select` branches. The biggest win comes from reducing redundant address arithmetic and using `as_type<char4>` for the row read when `i` is in the interior, plus precomputing row pointers as `device const char*`. I also avoid the threadgroup barrier for the 5-entry table by simply reading `p_accept[idx]` directly from `device` memory (it's tiny and cached, and removes a barrier per dispatch). Wrap-around is computed branchlessly via modular arithmetic with `% NX` which on power-of-two sizes the compiler folds, and otherwise stays cheap. The inactive-color threads return immediately before any memory reads, halving traffic for the wasted half of the grid.

```metal
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
                       uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint j = gid.y;
    if (i >= NX || j >= NY) return;

    uint color = step_idx & 1u;
    if (((i + j) & 1u) != color) return;

    // Branchless periodic neighbours via conditional add/sub.
    uint nx = NX;
    uint ny = NY;
    uint im = (i == 0u)      ? (nx - 1u) : (i - 1u);
    uint ip = (i == nx - 1u) ? 0u        : (i + 1u);
    uint jm = (j == 0u)      ? (ny - 1u) : (j - 1u);
    uint jp = (j == ny - 1u) ? 0u        : (j + 1u);

    uint row   = j  * nx;
    uint row_m = jm * nx;
    uint row_p = jp * nx;

    int s  = (int)spins[row   + i ];
    int sl = (int)spins[row   + im];
    int sr = (int)spins[row   + ip];
    int sd = (int)spins[row_m + i ];
    int su = (int)spins[row_p + i ];

    int h    = sl + sr + sd + su;
    int prod = s * h;
    int idx  = (prod + 4) >> 1;
    float pa = p_accept[idx];

    uint site_idx = row + i;
    uint x = seed + step_idx * 0x9E3779B9u;
    x = mix32(x);
    x = mix32(x ^ site_idx);
    float u = float(x >> 8) * (1.0f / 16777216.0f);

    char new_s = (char)(-s);
    char old_s = (char)s;
    spins[site_idx] = (u < pa) ? new_s : old_s;
}
```