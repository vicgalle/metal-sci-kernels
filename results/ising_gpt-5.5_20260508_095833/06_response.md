This version avoids the divergent checkerboard early-exit: all lanes execute the read/RNG path and only the final write is predicated by color and acceptance. On SIMD hardware this improves lane utilization versus running the heavy path with half the lanes masked. It also replaces `s*h` with an equivalent same-neighbour count and uses a fast power-of-two periodic-index path for the benchmark sizes, with a general fallback.

```metal
#include <metal_stdlib>
using namespace metal;

inline uint mix32(uint x) {
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    return x ^ (x >> 16);
}

inline void ising_update_predicated(device       char  *spins,
                                    device const float *p_accept,
                                    uint site_idx,
                                    uint left_idx,
                                    uint right_idx,
                                    uint up_idx,
                                    uint down_idx,
                                    uint color_bit,
                                    uint step_idx,
                                    uint seed) {
    const char s = spins[site_idx];

    // Equivalent to (s * (sl + sr + su + sd) + 4) / 2 for spins in {-1,+1}:
    // it is simply the number of neighbours equal to the center spin.
    const uint idx =
        ((spins[left_idx]  == s) ? 1u : 0u) +
        ((spins[right_idx] == s) ? 1u : 0u) +
        ((spins[up_idx]    == s) ? 1u : 0u) +
        ((spins[down_idx]  == s) ? 1u : 0u);

    const float pa = p_accept[idx];

    uint x = seed + step_idx * 0x9E3779B9u;
    x = mix32(x);
    x = mix32(x ^ site_idx);

    const float u = float(x >> 8) * (1.0f / 16777216.0f);
    const uint accept_bit = (u < pa) ? 1u : 0u;

    if ((color_bit & accept_bit) != 0u) {
        spins[site_idx] = (char)(-((int)s));
    }
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

    const uint site_idx = j * nx + i;
    const uint color_bit = 1u ^ ((i ^ j ^ step_idx) & 1u);

    const uint pow2_test = (nx & (nx - 1u)) | (ny & (ny - 1u));

    if (pow2_test == 0u) {
        const uint total  = nx * ny;
        const uint mask_x = nx - 1u;
        const uint mask_t = total - 1u;

        const uint row_base  = site_idx & ~mask_x;
        const uint left_idx  = row_base | ((site_idx - 1u) & mask_x);
        const uint right_idx = row_base | ((site_idx + 1u) & mask_x);
        const uint up_idx    = (site_idx - nx) & mask_t;
        const uint down_idx  = (site_idx + nx) & mask_t;

        ising_update_predicated(spins, p_accept,
                                site_idx, left_idx, right_idx, up_idx, down_idx,
                                color_bit, step_idx, seed);
    } else {
        const uint total = nx * ny;

        const uint left_idx  = (i == 0u)        ? (site_idx + nx - 1u)    : (site_idx - 1u);
        const uint right_idx = (i + 1u == nx)   ? (site_idx + 1u - nx)    : (site_idx + 1u);
        const uint up_idx    = (j == 0u)        ? (site_idx + total - nx) : (site_idx - nx);
        const uint down_idx  = (j + 1u == ny)   ? (site_idx + nx - total) : (site_idx + nx);

        ising_update_predicated(spins, p_accept,
                                site_idx, left_idx, right_idx, up_idx, down_idx,
                                color_bit, step_idx, seed);
    }
}
```