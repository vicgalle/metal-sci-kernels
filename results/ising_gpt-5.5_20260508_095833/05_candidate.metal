#include <metal_stdlib>
using namespace metal;

inline uint mix32(uint x) {
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    return x ^ (x >> 16);
}

inline uint ising_rand_u32(uint seed, uint step_idx, uint site_idx) {
    uint x = seed + step_idx * 0x9E3779B9u;
    x = mix32(x);
    return mix32(x ^ site_idx);
}

kernel void ising_step(device       char  *spins    [[buffer(0)]],
                       device const float *p_accept [[buffer(1)]],
                       constant uint  &NX           [[buffer(2)]],
                       constant uint  &NY           [[buffer(3)]],
                       constant uint  &step_idx     [[buffer(4)]],
                       constant uint  &seed         [[buffer(5)]],
                       uint2 gid [[thread_position_in_grid]]) {
    const uint nx = NX;
    const uint ny = NY;
    const uint i = gid.x;
    const uint j = gid.y;

    if (i >= nx || j >= ny) return;

    if (((i ^ j ^ step_idx) & 1u) != 0u) return;

    const uint site_idx = j * nx + i;

    int s;
    int h;

    if (i > 0u && (i + 1u) < nx && j > 0u && (j + 1u) < ny) {
        s = int(spins[site_idx]);
        h = int(spins[site_idx - 1u]) +
            int(spins[site_idx + 1u]) +
            int(spins[site_idx - nx]) +
            int(spins[site_idx + nx]);
    } else {
        const uint total = nx * ny;

        const uint left_idx  = (i == 0u)      ? (site_idx + nx - 1u)    : (site_idx - 1u);
        const uint right_idx = (i + 1u == nx) ? (site_idx + 1u - nx)    : (site_idx + 1u);
        const uint up_idx    = (j == 0u)      ? (site_idx + total - nx) : (site_idx - nx);
        const uint down_idx  = (j + 1u == ny) ? (site_idx + nx - total) : (site_idx + nx);

        s = int(spins[site_idx]);
        h = int(spins[left_idx]) +
            int(spins[right_idx]) +
            int(spins[up_idx]) +
            int(spins[down_idx]);
    }

    const int prod = s * h;

    if (prod <= 0) {
        spins[site_idx] = char(-s);
        return;
    }

    const uint pidx = uint(prod + 4) >> 1;
    const float pa = p_accept[pidx];

    const uint bits = ising_rand_u32(seed, step_idx, site_idx);
    const float u = float(bits >> 8) * (1.0f / 16777216.0f);

    if (u < pa) {
        spins[site_idx] = char(-s);
    }
}