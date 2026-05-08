#include <metal_stdlib>
using namespace metal;

inline uint mix32(uint x) {
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    return x ^ (x >> 16);
}

inline uint accept_idx_pm1(int s, int l, int r, int u, int d) {
    // For int spins in {-1,+1}, differing spins xor to -2, whose top bit is 1.
    const uint diff =
        ((uint)(l ^ s) >> 31) +
        ((uint)(r ^ s) >> 31) +
        ((uint)(u ^ s) >> 31) +
        ((uint)(d ^ s) >> 31);
    return 4u - diff;
}

inline int3 load_horiz4_pow2(device char *spins,
                             uint site_idx,
                             uint row_base,
                             uint i,
                             uint nx,
                             uint mask_x) {
    // Interior: one unaligned packed 4-byte load supplies left, center, right.
    // Require i+2<nx so the extra fourth byte remains in-row/in-buffer.
    if ((i > 0u) && (i + 2u < nx)) {
        device const packed_char4 *p =
            (device const packed_char4 *)(spins + site_idx - 1u);
        const packed_char4 v = *p;
        return int3((int)v.x, (int)v.y, (int)v.z); // l, s, r
    } else {
        const int s = (int)spins[site_idx];
        const int l = (int)spins[row_base | ((i - 1u) & mask_x)];
        const int r = (int)spins[row_base | ((i + 1u) & mask_x)];
        return int3(l, s, r);
    }
}

inline int2 load_vert_xor16_pow2(device char *spins,
                                 uint site_idx,
                                 uint nx,
                                 uint mask_t,
                                 int s) {
    const uint up_idx   = (site_idx - nx) & mask_t;
    const uint down_idx = (site_idx + nx) & mask_t;

    // Common 16x16 2-D groups map lane^16 to the same x in the adjacent row.
    // If that is not true for the host's group shape, fall back to global loads.
    const int  pair_s   = simd_shuffle_xor(s, ushort(16));
    const uint pair_idx = simd_shuffle_xor(site_idx, ushort(16));

    const bool pair_is_up   = (pair_idx + nx == site_idx);
    const bool pair_is_down = (site_idx + nx == pair_idx);

    int u;
    int d;

    if (pair_is_up || pair_is_down) {
        const uint other_idx = pair_is_up ? down_idx : up_idx;
        const int other = (int)spins[other_idx];

        u = pair_is_up ? pair_s : other;
        d = pair_is_up ? other  : pair_s;
    } else {
        u = (int)spins[up_idx];
        d = (int)spins[down_idx];
    }

    return int2(u, d);
}

inline void update_active_loaded(device       char  *spins,
                                 device const float *p_accept,
                                 uint site_idx,
                                 int s,
                                 int l,
                                 int r,
                                 int u,
                                 int d,
                                 uint rng_base) {
    const uint idx = accept_idx_pm1(s, l, r, u, d);
    const float pa = p_accept[idx];

    const uint bits = mix32(rng_base ^ site_idx);
    const float draw = float(bits >> 8) * (1.0f / 16777216.0f);

    if (draw < pa) {
        spins[site_idx] = (char)(-s);
    }
}

inline void update_pred_loaded(device       char  *spins,
                               device const float *p_accept,
                               uint site_idx,
                               int s,
                               int l,
                               int r,
                               int u,
                               int d,
                               uint active_bit,
                               uint rng_base) {
    const uint idx = accept_idx_pm1(s, l, r, u, d);
    const float pa = p_accept[idx];

    const uint bits = mix32(rng_base ^ site_idx);
    const float draw = float(bits >> 8) * (1.0f / 16777216.0f);
    const uint accept_bit = (draw < pa) ? 1u : 0u;

    if ((active_bit & accept_bit) != 0u) {
        spins[site_idx] = (char)(-s);
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

    // Small benchmark: do not spend RNG/load work on inactive checkerboard sites.
    if (nx == 256u && ny == 256u) {
        if (parity != 0u) return;

        const uint site_idx = (j << 8) | i;
        const uint row_base = j << 8;
        const uint rng_base = mix32(seed + step_idx * 0x9E3779B9u);

        const int3 h = load_horiz4_pow2(spins, site_idx, row_base,
                                         i, 256u, 255u);
        const int u = (int)spins[(site_idx - 256u) & 65535u];
        const int d = (int)spins[(site_idx + 256u) & 65535u];

        update_active_loaded(spins, p_accept,
                             site_idx, h.y, h.x, h.z, u, d, rng_base);
        return;
    }

    // Large power-of-two benchmark sizes.
    if (nx == 1024u && ny == 1024u) {
        const uint site_idx = (j << 10) | i;
        const uint row_base = j << 10;
        const uint rng_base = mix32(seed + step_idx * 0x9E3779B9u);

        const int3 h = load_horiz4_pow2(spins, site_idx, row_base,
                                         i, 1024u, 1023u);
        const int2 v = load_vert_xor16_pow2(spins, site_idx,
                                             1024u, 1048575u, h.y);

        const uint active_bit = 1u ^ parity;

        update_pred_loaded(spins, p_accept,
                           site_idx, h.y, h.x, h.z, v.x, v.y,
                           active_bit, rng_base);
        return;
    }

    if (nx == 2048u && ny == 2048u) {
        const uint site_idx = (j << 11) | i;
        const uint row_base = j << 11;
        const uint rng_base = mix32(seed + step_idx * 0x9E3779B9u);

        const int3 h = load_horiz4_pow2(spins, site_idx, row_base,
                                         i, 2048u, 2047u);
        const int2 v = load_vert_xor16_pow2(spins, site_idx,
                                             2048u, 4194303u, h.y);

        const uint active_bit = 1u ^ parity;

        update_pred_loaded(spins, p_accept,
                           site_idx, h.y, h.x, h.z, v.x, v.y,
                           active_bit, rng_base);
        return;
    }

    // Generic fallback.
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

    const int s = (int)spins[site_idx];
    const int l = (int)spins[left_idx];
    const int r = (int)spins[right_idx];
    const int u = (int)spins[up_idx];
    const int d = (int)spins[down_idx];

    const uint rng_base = mix32(seed + step_idx * 0x9E3779B9u);

    if (small_lattice) {
        update_active_loaded(spins, p_accept,
                             site_idx, s, l, r, u, d, rng_base);
    } else {
        const uint active_bit = 1u ^ parity;
        update_pred_loaded(spins, p_accept,
                           site_idx, s, l, r, u, d,
                           active_bit, rng_base);
    }
}