#include <metal_stdlib>
using namespace metal;

inline uint mix32(uint x) {
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    return x ^ (x >> 16);
}

inline uint rand_u32(uint seed, uint step_idx, uint site_idx) {
    uint x = seed + step_idx * 0x9E3779B9u;
    x = mix32(x);
    x = mix32(x ^ site_idx);
    return x;
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

    if (i >= nx || j >= ny) return;

    if (((i ^ j ^ step_idx) & 1u) != 0u) return;

    const uint row = j * nx;
    const uint site_idx = row + i;

    const int s = int(spins[site_idx]);

    int h;
    if (i > 0u && (i + 1u) < nx && j > 0u && (j + 1u) < ny) {
        h = int(spins[site_idx - 1u]) +
            int(spins[site_idx + 1u]) +
            int(spins[site_idx - nx]) +
            int(spins[site_idx + nx]);
    } else {
        const uint im = (i == 0u) ? (nx - 1u) : (i - 1u);
        const uint ip = ((i + 1u) == nx) ? 0u : (i + 1u);
        const uint row_m = (j == 0u) ? ((ny - 1u) * nx) : (row - nx);
        const uint row_p = ((j + 1u) == ny) ? 0u : (row + nx);

        h = int(spins[row + im]) +
            int(spins[row + ip]) +
            int(spins[row_m + i]) +
            int(spins[row_p + i]);
    }

    const int prod = s * h;
    const uint pidx = uint(prod + 4) >> 1;
    const float pa = p_accept[pidx];

    if (pa >= 1.0f) {
        spins[site_idx] = char(-s);
        return;
    }

    if (pa <= 0.0f) {
        return;
    }

    const uint bits = rand_u32(seed, step_idx, site_idx);
    const float u = float(bits >> 8) * (1.0f / 16777216.0f);

    if (u < pa) {
        spins[site_idx] = char(-s);
    }
}