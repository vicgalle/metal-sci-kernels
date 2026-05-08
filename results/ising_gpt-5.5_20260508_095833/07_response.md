Applies a hybrid checkerboard strategy: small lattices keep early-exit for wrong-color threads to minimize wasted launch work, while 1024²/2048² use predicated full-grid execution to avoid severe SIMD under-utilization from alternating active lanes. Power-of-two specializations replace dynamic boundary arithmetic with masks/shifts for the benchmark sizes, reducing integer overhead versus the incumbent while preserving bit-exact RNG and acceptance behavior.

```metal
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

inline uint accept_index(device char *spins,
                         uint site_idx,
                         uint left_idx,
                         uint right_idx,
                         uint up_idx,
                         uint down_idx) {
    const char s = spins[site_idx];

    // For spins in {-1,+1}, (s*h + 4)/2 is exactly the number of
    // neighbours equal to s.
    return ((spins[left_idx]  == s) ? 1u : 0u) +
           ((spins[right_idx] == s) ? 1u : 0u) +
           ((spins[up_idx]    == s) ? 1u : 0u) +
           ((spins[down_idx]  == s) ? 1u : 0u);
}

inline void update_active(device       char  *spins,
                          device const float *p_accept,
                          uint site_idx,
                          uint left_idx,
                          uint right_idx,
                          uint up_idx,
                          uint down_idx,
                          uint step_idx,
                          uint seed) {
    const char s = spins[site_idx];
    const uint idx =
        ((spins[left_idx]  == s) ? 1u : 0u) +
        ((spins[right_idx] == s) ? 1u : 0u) +
        ((spins[up_idx]    == s) ? 1u : 0u) +
        ((spins[down_idx]  == s) ? 1u : 0u);

    const float pa = p_accept[idx];

    const uint bits = rand_u32(seed, step_idx, site_idx);
    const float u = float(bits >> 8) * (1.0f / 16777216.0f);

    if (u < pa) {
        spins[site_idx] = (char)(-((int)s));
    }
}

inline void update_predicated(device       char  *spins,
                              device const float *p_accept,
                              uint site_idx,
                              uint left_idx,
                              uint right_idx,
                              uint up_idx,
                              uint down_idx,
                              uint active_bit,
                              uint step_idx,
                              uint seed) {
    const char s = spins[site_idx];
    const uint idx =
        ((spins[left_idx]  == s) ? 1u : 0u) +
        ((spins[right_idx] == s) ? 1u : 0u) +
        ((spins[up_idx]    == s) ? 1u : 0u) +
        ((spins[down_idx]  == s) ? 1u : 0u);

    const float pa = p_accept[idx];

    const uint bits = rand_u32(seed, step_idx, site_idx);
    const float u = float(bits >> 8) * (1.0f / 16777216.0f);
    const uint accept_bit = (u < pa) ? 1u : 0u;

    if ((active_bit & accept_bit) != 0u) {
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

    const uint parity = (i ^ j ^ step_idx) & 1u;

    // 256x256: small benchmark is launch/overhead sensitive, so avoid doing
    // RNG/load work for the inactive checkerboard color.
    if (nx == 256u && ny == 256u) {
        if (parity != 0u) return;

        const uint site_idx = (j << 8) | i;
        const uint row_base = j << 8;

        const uint left_idx  = row_base | ((i - 1u) & 255u);
        const uint right_idx = row_base | ((i + 1u) & 255u);
        const uint up_idx    = (site_idx - 256u) & 65535u;
        const uint down_idx  = (site_idx + 256u) & 65535u;

        update_active(spins, p_accept,
                      site_idx, left_idx, right_idx, up_idx, down_idx,
                      step_idx, seed);
        return;
    }

    // Large power-of-two benchmark sizes: keep all SIMD lanes busy and only
    // predicate the final write for the inactive checkerboard color.
    if (nx == 1024u && ny == 1024u) {
        const uint site_idx = (j << 10) | i;
        const uint row_base = j << 10;

        const uint left_idx  = row_base | ((i - 1u) & 1023u);
        const uint right_idx = row_base | ((i + 1u) & 1023u);
        const uint up_idx    = (site_idx - 1024u) & 1048575u;
        const uint down_idx  = (site_idx + 1024u) & 1048575u;

        const uint active_bit = 1u ^ parity;

        update_predicated(spins, p_accept,
                          site_idx, left_idx, right_idx, up_idx, down_idx,
                          active_bit, step_idx, seed);
        return;
    }

    if (nx == 2048u && ny == 2048u) {
        const uint site_idx = (j << 11) | i;
        const uint row_base = j << 11;

        const uint left_idx  = row_base | ((i - 1u) & 2047u);
        const uint right_idx = row_base | ((i + 1u) & 2047u);
        const uint up_idx    = (site_idx - 2048u) & 4194303u;
        const uint down_idx  = (site_idx + 2048u) & 4194303u;

        const uint active_bit = 1u ^ parity;

        update_predicated(spins, p_accept,
                          site_idx, left_idx, right_idx, up_idx, down_idx,
                          active_bit, step_idx, seed);
        return;
    }

    // Generic fallback.  Use early-exit on small lattices and predicated
    // full-grid work on larger lattices, mirroring the specialized policy.
    const uint site_idx = j * nx + i;
    const uint total = nx * ny;
    const bool small_lattice = (total <= 262144u);

    if (small_lattice && parity != 0u) return;

    uint left_idx;
    uint right_idx;
    uint up_idx;
    uint down_idx;

    const uint pow2_test = (nx & (nx - 1u)) | (ny & (ny - 1u));

    if (pow2_test == 0u) {
        const uint mask_x = nx - 1u;
        const uint mask_t = total - 1u;
        const uint row_base = site_idx & ~mask_x;

        left_idx  = row_base | ((i - 1u) & mask_x);
        right_idx = row_base | ((i + 1u) & mask_x);
        up_idx    = (site_idx - nx) & mask_t;
        down_idx  = (site_idx + nx) & mask_t;
    } else {
        left_idx  = (i == 0u)        ? (site_idx + nx - 1u)    : (site_idx - 1u);
        right_idx = (i + 1u == nx)   ? (site_idx + 1u - nx)    : (site_idx + 1u);
        up_idx    = (j == 0u)        ? (site_idx + total - nx) : (site_idx - nx);
        down_idx  = (j + 1u == ny)   ? (site_idx + nx - total) : (site_idx + nx);
    }

    if (small_lattice) {
        update_active(spins, p_accept,
                      site_idx, left_idx, right_idx, up_idx, down_idx,
                      step_idx, seed);
    } else {
        const uint active_bit = 1u ^ parity;
        update_predicated(spins, p_accept,
                          site_idx, left_idx, right_idx, up_idx, down_idx,
                          active_bit, step_idx, seed);
    }
}
```