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

    uint s_step = step_idx;
    if (((i + j) & 1u) != (s_step & 1u)) return;

    uint site_idx  = j * nx + i;
    uint grid_size = nx * ny;

    // Manually hoist uniform RNG calculations.
    uint base_x = seed + s_step * 0x9E3779B9u;
    base_x = (base_x ^ (base_x >> 16)) * 0x85EBCA6Bu;
    base_x = (base_x ^ (base_x >> 13)) * 0xC2B2AE35u;
    base_x = base_x ^ (base_x >> 16);

    // Compute periodic neighbor indices without multiplication.
    uint left_idx  = select(site_idx - 1u, site_idx + nx - 1u, i == 0u);
    uint right_idx = select(site_idx + 1u, site_idx - nx + 1u, i == nx - 1u);
    uint up_idx    = select(site_idx - nx, site_idx + grid_size - nx, j == 0u);
    uint down_idx  = select(site_idx + nx, i, j == ny - 1u);

    device const char* cspins = (device const char*)spins;

    int s  = cspins[site_idx];
    int sl = cspins[left_idx];
    int sr = cspins[right_idx];
    int su = cspins[up_idx];
    int sd = cspins[down_idx];

    int h = sl + sr + su + sd;
    int prod = s * h;

    int idx  = (prod + 4) >> 1;
    float pa = p_accept[idx];

    uint x = base_x ^ site_idx;
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    x = x ^ (x >> 16);

    float u = float(x >> 8) * (1.0f / 16777216.0f);

    if (u < pa) {
        spins[site_idx] = (char)(-s);
    }
}