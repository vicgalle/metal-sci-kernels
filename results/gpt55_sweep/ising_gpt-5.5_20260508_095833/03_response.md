This version avoids the slow power-of-two runtime path and adds two optimizations: deterministic early acceptance for `prod <= 0` (where Metropolis probability is exactly 1), skipping the expensive RNG, and a large-lattice SIMD shuffle path that lets wrong-color lanes preload their own spin so active lanes can reuse adjacent horizontal and common 16-wide vertical neighbors instead of issuing global loads. Fallback loads preserve correctness at boundaries and nonmatching SIMD layouts.

```metal
#include <metal_stdlib>
using namespace metal;

inline uint mix32(uint x) {
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    return x ^ (x >> 16);
}

inline void metropolis_update(device       char  *spins,
                              device const float *p_accept,
                              uint site_idx,
                              int s,
                              int h,
                              uint step_idx,
                              uint seed) {
    const int prod = s * h;

    // For Metropolis Ising, p_accept[0..2] are exactly 1.0f, and
    // u is always < 1, so these flips are deterministic.
    if (prod <= 0) {
        spins[site_idx] = (char)(-s);
        return;
    }

    const uint pidx = uint(prod + 4) >> 1;
    const float pa = p_accept[pidx];

    uint x = seed + step_idx * 0x9E3779B9u;
    x = mix32(x);
    x = mix32(x ^ site_idx);

    const float u = float(x >> 8) * (1.0f / 16777216.0f);

    if (u < pa) {
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
    const uint i = gid.x;
    const uint j = gid.y;
    const uint nx = NX;
    const uint ny = NY;
    const uint step = step_idx;

    if (i >= nx || j >= ny) return;

    // Keep the small case lean; dispatch overhead dominates there and the
    // scalar path avoids extra SIMD shuffle work on inactive checkerboard lanes.
    if (nx < 1024u || ny < 1024u) {
        if (((i ^ j ^ step) & 1u) != 0u) return;

        const uint row = j * nx;
        const uint site_idx = row + i;

        const uint im = (i == 0u)        ? (nx - 1u) : (i - 1u);
        const uint ip = (i == nx - 1u)   ? 0u        : (i + 1u);
        const uint jm = (j == 0u)        ? (ny - 1u) : (j - 1u);
        const uint jp = (j == ny - 1u)   ? 0u        : (j + 1u);

        const int s = int(spins[site_idx]);
        const int h = int(spins[row + im])      +
                      int(spins[row + ip])      +
                      int(spins[jm * nx + i])   +
                      int(spins[jp * nx + i]);

        metropolis_update(spins, p_accept, site_idx, s, h, step, seed);
        return;
    }

    // Large-lattice path: all lanes load their center spin before the color
    // test. Wrong-color lanes then provide neighbor values through SIMD shuffles.
    const uint row = j * nx;
    const uint site_idx = row + i;
    const uint total = nx * ny;

    const uint one = 1u;
    const uint lane = simd_prefix_exclusive_sum(one);
    const uint simd_count = simd_sum(one);

    const int s_center = int(spins[site_idx]);

    const ushort d1 = 1;
    const ushort d16 = 16;

    const int  s_l1    = simd_shuffle_up(s_center, d1);
    const int  s_r1    = simd_shuffle_down(s_center, d1);
    const uint site_l1 = simd_shuffle_up(site_idx, d1);
    const uint site_r1 = simd_shuffle_down(site_idx, d1);

    const int  s_u16    = simd_shuffle_up(s_center, d16);
    const int  s_d16    = simd_shuffle_down(s_center, d16);
    const uint site_u16 = simd_shuffle_up(site_idx, d16);
    const uint site_d16 = simd_shuffle_down(site_idx, d16);

    if (((i ^ j ^ step) & 1u) != 0u) return;

    int sl;
    if ((lane >= 1u) && (i > 0u) && (site_l1 == site_idx - 1u)) {
        sl = s_l1;
    } else {
        const uint left_idx = (i == 0u) ? (site_idx + nx - 1u) : (site_idx - 1u);
        sl = int(spins[left_idx]);
    }

    int sr;
    if ((lane + 1u < simd_count) && (i + 1u < nx) && (site_r1 == site_idx + 1u)) {
        sr = s_r1;
    } else {
        const uint right_idx = (i == nx - 1u) ? (site_idx + 1u - nx) : (site_idx + 1u);
        sr = int(spins[right_idx]);
    }

    int su;
    if ((lane >= 16u) && (j > 0u) && (site_u16 == site_idx - nx)) {
        su = s_u16;
    } else {
        const uint up_idx = (j == 0u) ? (site_idx + total - nx) : (site_idx - nx);
        su = int(spins[up_idx]);
    }

    int sd;
    if ((lane + 16u < simd_count) && (j + 1u < ny) && (site_d16 == site_idx + nx)) {
        sd = s_d16;
    } else {
        const uint down_idx = (j == ny - 1u) ? (site_idx + nx - total) : (site_idx + nx);
        sd = int(spins[down_idx]);
    }

    const int h = sl + sr + su + sd;
    metropolis_update(spins, p_accept, site_idx, s_center, h, step, seed);
}
```