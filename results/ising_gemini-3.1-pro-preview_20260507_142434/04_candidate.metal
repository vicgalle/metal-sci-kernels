#include <metal_stdlib>
using namespace metal;

kernel void ising_step(device       char  *spins    [[buffer(0)]],
                       device const float *p_accept [[buffer(1)]],
                       constant uint  &NX           [[buffer(2)]],
                       constant uint  &NY           [[buffer(3)]],
                       constant uint  &step_idx     [[buffer(4)]],
                       constant uint  &seed         [[buffer(5)]],
                       uint2 gid [[thread_position_in_grid]]) {
    
    // 1. Uniform precomputations. Hoisted to the very top to guarantee 
    // execution on the scalar ALU before vector processing begins.
    float p0 = p_accept[0];
    float p1 = p_accept[1];
    float p2 = p_accept[2];
    float p3 = p_accept[3];
    float p4 = p_accept[4];

    uint base_x = seed + step_idx * 0x9E3779B9u;
    base_x = (base_x ^ (base_x >> 16)) * 0x85EBCA6Bu;
    base_x = (base_x ^ (base_x >> 13)) * 0xC2B2AE35u;
    base_x = base_x ^ (base_x >> 16);

    // 2. Thread bounds and checkerboard gating
    uint i = gid.x;
    uint j = gid.y;
    if (i >= NX || j >= NY) return;

    uint color = step_idx & 1u;
    if (((i + j) & 1u) != color) return;

    // 3. Periodic boundary conditions
    uint im = (i == 0u)      ? (NX - 1u) : (i - 1u);
    uint ip = (i == NX - 1u) ? 0u        : (i + 1u);
    uint jm = (j == 0u)      ? (NY - 1u) : (j - 1u);
    uint jp = (j == NY - 1u) ? 0u        : (j + 1u);

    // 4. Shared row offset calculations
    uint j_NX  = j * NX;
    uint jm_NX = jm * NX;
    uint jp_NX = jp * NX;

    int s  = (int)spins[j_NX  + i ];
    int sl = (int)spins[j_NX  + im];
    int sr = (int)spins[j_NX  + ip];
    int sd = (int)spins[jm_NX + i ];
    int su = (int)spins[jp_NX + i ];

    // 5. Energy computation
    int h    = sl + sr + sd + su;
    int prod = s * h;
    uint idx = (uint)(prod + 4) >> 1u;

    // Register-based select replaces constant-cache gather loads
    float pa = (idx == 0u) ? p0 :
               (idx == 1u) ? p1 :
               (idx == 2u) ? p2 :
               (idx == 3u) ? p3 : p4;

    // 6. Branchless RNG
    uint site_idx = j_NX + i;
    uint x = base_x ^ site_idx;
    
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    x = x ^ (x >> 16);

    float u = float(x >> 8) * (1.0f / 16777216.0f);

    // 7. Conditional writeback
    if (u < pa) {
        spins[site_idx] = (char)(-s);
    }
}