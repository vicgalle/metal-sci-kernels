I add a threadgroup-memory stencil tile for the large 1024²/2048² cases, using the same synchronization point to share the pass-uniform first RNG mix within the threadgroup. This should reduce repeated global neighbor loads while preserving the exact per-site RNG formula; non-tileable threadgroup shapes fall back to the prior specialized paths.

```metal
#include <metal_stdlib>
using namespace metal;

#define TILE_STRIDE 34u
#define TILE_ELEMS  1156

inline uint mix32(uint x) {
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    return x ^ (x >> 16);
}

inline uint rng_initial_base(uint seed, uint step_idx) {
    return mix32(seed + step_idx * 0x9E3779B9u);
}

inline void update_from_values(device       char  *spins,
                               device const float *p_accept,
                               uint site_idx,
                               char s,
                               char l,
                               char r,
                               char u,
                               char d,
                               uint active_bit,
                               uint rng_base) {
    const uint bits = mix32(rng_base ^ site_idx);

    const uint idx =
        ((l == s) ? 1u : 0u) +
        ((r == s) ? 1u : 0u) +
        ((u == s) ? 1u : 0u) +
        ((d == s) ? 1u : 0u);

    const float pa = p_accept[idx];
    const float draw = float(bits >> 8) * (1.0f / 16777216.0f);
    const uint accept_bit = (draw < pa) ? 1u : 0u;

    if ((active_bit & accept_bit) != 0u) {
        spins[site_idx] = (char)(-((int)s));
    }
}

inline void update_global_base(device       char  *spins,
                               device const float *p_accept,
                               uint site_idx,
                               uint left_idx,
                               uint right_idx,
                               uint up_idx,
                               uint down_idx,
                               uint active_bit,
                               uint rng_base) {
    const char s = spins[site_idx];
    const char l = spins[left_idx];
    const char r = spins[right_idx];
    const char u = spins[up_idx];
    const char d = spins[down_idx];

    update_from_values(spins, p_accept, site_idx, s, l, r, u, d,
                       active_bit, rng_base);
}

kernel void ising_step(device       char  *spins    [[buffer(0)]],
                       device const float *p_accept [[buffer(1)]],
                       constant uint  &NX           [[buffer(2)]],
                       constant uint  &NY           [[buffer(3)]],
                       constant uint  &step_idx     [[buffer(4)]],
                       constant uint  &seed         [[buffer(5)]],
                       uint2 gid [[thread_position_in_grid]],
                       uint2 tid [[thread_position_in_threadgroup]],
                       uint2 tpg [[threads_per_threadgroup]]) {
    const uint nx = NX;
    const uint ny = NY;
    const uint i = gid.x;
    const uint j = gid.y;
    const bool in_bounds = (i < nx) && (j < ny);

    threadgroup char tile[TILE_ELEMS];
    threadgroup uint tg_rng_base;

    const uint tg_count = tpg.x * tpg.y;
    const bool big_pow2_square =
        ((nx == 1024u && ny == 1024u) ||
         (nx == 2048u && ny == 2048u));

    const bool use_tile =
        big_pow2_square &&
        (tpg.x <= 32u) &&
        (tpg.y <= 32u) &&
        (tg_count >= 64u);

    if (use_tile) {
        uint site_idx = 0u;
        const uint shift = (nx == 1024u) ? 10u : 11u;
        const uint mask_x = nx - 1u;
        const uint mask_y = ny - 1u;

        const uint lx = tid.x;
        const uint ly = tid.y;
        const uint center = (ly + 1u) * TILE_STRIDE + (lx + 1u);

        if (in_bounds) {
            const uint row = j << shift;
            site_idx = row | i;

            tile[center] = spins[site_idx];

            if (lx == 0u) {
                tile[(ly + 1u) * TILE_STRIDE] =
                    spins[row | ((i - 1u) & mask_x)];
            }

            if ((lx + 1u == tpg.x) || (i + 1u == nx)) {
                tile[(ly + 1u) * TILE_STRIDE + (lx + 2u)] =
                    spins[row | ((i + 1u) & mask_x)];
            }

            if (ly == 0u) {
                tile[lx + 1u] =
                    spins[(((j - 1u) & mask_y) << shift) | i];
            }

            if ((ly + 1u == tpg.y) || (j + 1u == ny)) {
                tile[(ly + 2u) * TILE_STRIDE + (lx + 1u)] =
                    spins[(((j + 1u) & mask_y) << shift) | i];
            }
        }

        if (tid.x == 0u && tid.y == 0u) {
            tg_rng_base = rng_initial_base(seed, step_idx);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (!in_bounds) return;

        const uint active_bit = 1u ^ ((i ^ j ^ step_idx) & 1u);

        const char s = tile[center];
        const char l = tile[center - 1u];
        const char r = tile[center + 1u];
        const char u = tile[center - TILE_STRIDE];
        const char d = tile[center + TILE_STRIDE];

        update_from_values(spins, p_accept, site_idx, s, l, r, u, d,
                           active_bit, tg_rng_base);
        return;
    }

    const bool use_tg_rng_base = big_pow2_square && (tg_count >= 128u);
    if (use_tg_rng_base) {
        if (tid.x == 0u && tid.y == 0u) {
            tg_rng_base = rng_initial_base(seed, step_idx);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!in_bounds) return;

    const uint parity = (i ^ j ^ step_idx) & 1u;

    if (nx == 256u && ny == 256u) {
        if (parity != 0u) return;

        const uint site_idx = (j << 8) | i;
        const uint row_base = j << 8;

        const uint left_idx  = row_base | ((i - 1u) & 255u);
        const uint right_idx = row_base | ((i + 1u) & 255u);
        const uint up_idx    = (site_idx - 256u) & 65535u;
        const uint down_idx  = (site_idx + 256u) & 65535u;

        update_global_base(spins, p_accept,
                           site_idx, left_idx, right_idx, up_idx, down_idx,
                           1u, rng_initial_base(seed, step_idx));
        return;
    }

    if (nx == 1024u && ny == 1024u) {
        const uint site_idx = (j << 10) | i;
        const uint row_base = j << 10;

        const uint left_idx  = row_base | ((i - 1u) & 1023u);
        const uint right_idx = row_base | ((i + 1u) & 1023u);
        const uint up_idx    = (site_idx - 1024u) & 1048575u;
        const uint down_idx  = (site_idx + 1024u) & 1048575u;

        const uint active_bit = 1u ^ parity;
        const uint base = use_tg_rng_base ? tg_rng_base
                                          : rng_initial_base(seed, step_idx);

        update_global_base(spins, p_accept,
                           site_idx, left_idx, right_idx, up_idx, down_idx,
                           active_bit, base);
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
        const uint base = use_tg_rng_base ? tg_rng_base
                                          : rng_initial_base(seed, step_idx);

        update_global_base(spins, p_accept,
                           site_idx, left_idx, right_idx, up_idx, down_idx,
                           active_bit, base);
        return;
    }

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
        left_idx  = (i == 0u)      ? (site_idx + nx - 1u)    : (site_idx - 1u);
        right_idx = (i + 1u == nx) ? (site_idx + 1u - nx)    : (site_idx + 1u);
        up_idx    = (j == 0u)      ? (site_idx + total - nx) : (site_idx - nx);
        down_idx  = (j + 1u == ny) ? (site_idx + nx - total) : (site_idx + nx);
    }

    const uint active_bit = small_lattice ? 1u : (1u ^ parity);

    update_global_base(spins, p_accept,
                       site_idx, left_idx, right_idx, up_idx, down_idx,
                       active_bit, rng_initial_base(seed, step_idx));
}
```