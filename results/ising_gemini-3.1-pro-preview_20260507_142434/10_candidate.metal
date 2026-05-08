#include <metal_stdlib>
using namespace metal;

kernel void ising_step(device       char  *spins    [[buffer(0)]],
                       device const float *p_accept [[buffer(1)]],
                       constant uint  &NX           [[buffer(2)]],
                       constant uint  &NY           [[buffer(3)]],
                       constant uint  &step_idx     [[buffer(4)]],
                       constant uint  &seed         [[buffer(5)]],
                       uint2 gid [[thread_position_in_grid]]) {
    uint nx = NX;
    uint ny = NY;
    uint i = gid.x;
    uint j = gid.y;
    
    if (i >= nx || j >= ny) return;

    uint step = step_idx;
    
    // Grid-uniform base RNG computation; compiler hoists to scalar registers
    uint base_x = seed + step * 0x9E3779B9u;
    base_x = (base_x ^ (base_x >> 16)) * 0x85EBCA6Bu;
    base_x = (base_x ^ (base_x >> 13)) * 0xC2B2AE35u;
    base_x = base_x ^ (base_x >> 16);

    // Faster checkerboard early exit using XOR parity
    if ((i ^ j ^ step) & 1u) return;

    // Periodic boundary neighbors
    uint im = (i == 0u)      ? (nx - 1u) : (i - 1u);
    uint ip = (i == nx - 1u) ? 0u        : (i + 1u);
    uint jm = (j == 0u)      ? (ny - 1u) : (j - 1u);
    uint jp = (j == ny - 1u) ? 0u        : (j + 1u);

    // CSE row offsets for faster address generation
    uint j_nx  = j * nx;
    uint jm_nx = jm * nx;
    uint jp_nx = jp * nx;

    uint site = j_nx + i;

    // Direct L1-cached byte reads
    int s  = (int)spins[site];
    int sl = (int)spins[j_nx + im];
    int sr = (int)spins[j_nx + ip];
    int sd = (int)spins[jm_nx + i];
    int su = (int)spins[jp_nx + i];

    int h    = sl + sr + sd + su;
    int prod = s * h;
    int idx  = (prod + 4) >> 1; // replaces division

    float pa = p_accept[idx];

    // Second site-specific RNG mix
    uint x = base_x ^ site;
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    x = x ^ (x >> 16);

    // Exact bitwise float conversion
    float u = float(x >> 8) * (1.0f / 16777216.0f);

    if (u < pa) {
        spins[site] = (char)(-s);
    }
}