#include <metal_stdlib>
using namespace metal;

inline uint mix32(uint x) {
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    return x ^ (x >> 16);
}

kernel void ising_step(device       char  *spins    [[buffer(0)]],
                       device const float *p_accept [[buffer(1)]],
                       constant uint  &NX           [[buffer(2)]],
                       constant uint  &NY           [[buffer(3)]],
                       constant uint  &step_idx     [[buffer(4)]],
                       constant uint  &seed         [[buffer(5)]],
                       uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x;
    const uint j = gid.y;
    const uint nx = NX;
    const uint ny = NY;
    const uint step = step_idx;

    if (i >= nx || j >= ny) return;

    if (((i ^ j ^ step) & 1u) != 0u) return;

    const uint row = j * nx;
    const uint site_idx = row + i;
    const uint total = nx * ny;

    uint left_idx;
    uint right_idx;
    uint up_idx;
    uint down_idx;

    const bool pow2_dims = (((nx & (nx - 1u)) == 0u) &&
                            ((ny & (ny - 1u)) == 0u));

    if (pow2_dims) {
        const uint xmask = nx - 1u;
        const uint tmask = total - 1u;

        left_idx  = row + ((i - 1u) & xmask);
        right_idx = row + ((i + 1u) & xmask);
        up_idx    = (site_idx - nx) & tmask;
        down_idx  = (site_idx + nx) & tmask;
    } else {
        const uint nxm1 = nx - 1u;
        const uint nym1 = ny - 1u;

        left_idx  = (i == 0u)    ? (site_idx + nxm1)       : (site_idx - 1u);
        right_idx = (i == nxm1)  ? (site_idx - nxm1)       : (site_idx + 1u);
        up_idx    = (j == 0u)    ? (site_idx + total - nx) : (site_idx - nx);
        down_idx  = (j == nym1)  ? (site_idx + nx - total) : (site_idx + nx);
    }

    const int s = int(spins[site_idx]);

    const int h = int(spins[left_idx])  +
                  int(spins[right_idx]) +
                  int(spins[up_idx])    +
                  int(spins[down_idx]);

    const int prod = s * h;
    const uint pidx = uint(prod + 4) >> 1;
    const float pa = p_accept[pidx];

    uint x = seed + step * 0x9E3779B9u;
    x = mix32(x);
    x = mix32(x ^ site_idx);

    const float u = float(x >> 8) * (1.0f / 16777216.0f);

    if (u < pa) {
        spins[site_idx] = char(-s);
    }
}