## Task: ising

2D Ising model with checkerboard Metropolis updates and periodic boundary conditions. Spins are int8 in {-1, +1} stored row-major as `device char *spins[NY*NX]`.

One sub-pass updates one color of the checkerboard:
  color = step_idx & 1   (0 = (i+j) even, 1 = (i+j) odd)
The host dispatches this kernel 2 * n_sweeps times with step_idx = 0, 1, 2, ... so each full sweep is one red pass followed by one black pass. Within a sub-pass all updates are independent (the neighbours of a color-c site are all color 1-c, untouched).

For each color-matching site (i, j):
  h = spins[j,(i-1)%NX] + spins[j,(i+1)%NX]
    + spins[(j-1)%NY,i] + spins[(j+1)%NY,i]      in {-4,-2,0,2,4}
  prod = spins[j,i] * h                          in {-4,-2,0,2,4}
  pa = p_accept[(prod + 4) / 2]                  fp32
  draw uniform u in [0, 1) via the prescribed RNG
  if u < pa: spins[j,i] = -spins[j,i]

RNG (must be reproduced bit-exactly):
  inline uint mix32(uint x) {
      x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
      x = (x ^ (x >> 13)) * 0xC2B2AE35u;
      return x ^ (x >> 16);
  }
  uint x = seed + step_idx * 0x9E3779B9u;
  x = mix32(x);
  x = mix32(x ^ site_idx);            // site_idx = j * NX + i
  float u = float(x >> 8) * (1.0f / 16777216.0f);
Both the integer-to-float conversion and the multiply by
2^-24 are exact in fp32, so candidate kernels MUST use this
exact formula (or a provably-equivalent rearrangement).
The acceptance table p_accept[5] is precomputed by the host
in fp32 and read from buffer 1; do NOT call exp() on the GPU.


## Required kernel signature(s)

```
kernel void ising_step(device       char  *spins    [[buffer(0)]],
                       device const float *p_accept [[buffer(1)]],
                       constant uint  &NX           [[buffer(2)]],
                       constant uint  &NY           [[buffer(3)]],
                       constant uint  &step_idx     [[buffer(4)]],
                       constant uint  &seed         [[buffer(5)]],
                       uint2 gid [[thread_position_in_grid]]);

Grid is dispatched 2-D as `threadsPerGrid = (NX, NY)`, one thread per lattice site; guard with `if (i >= NX || j >= NY) return;`. Threads of the wrong color (`(i+j) & 1 != color`) MUST NOT mutate spins[j*NX + i] — they may early-exit or take the read path with a predicated write. The host will not shrink the dispatch if you process multiple sites per thread, so any reorganisation must keep the (NX, NY) grid shape and the bit-exact RNG/acceptance formula above.
```

## Your previous attempt

```metal
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
```

Result of previous attempt:
       256x256_100: correct, 3.74 ms, 3.5 GB/s (effective, 2 B/site/sweep) (1.8% of 200 GB/s)
      1024x1024_50: correct, 6.19 ms, 16.9 GB/s (effective, 2 B/site/sweep) (8.5% of 200 GB/s)
      2048x2048_25: correct, 11.74 ms, 17.9 GB/s (effective, 2 B/site/sweep) (8.9% of 200 GB/s)
  score (gmean of fraction): 0.0510

## Current best (incumbent)

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

Incumbent result:
       256x256_100: correct, 1.58 ms, 8.3 GB/s (effective, 2 B/site/sweep) (4.1% of 200 GB/s)
      1024x1024_50: correct, 3.36 ms, 31.2 GB/s (effective, 2 B/site/sweep) (15.6% of 200 GB/s)
      2048x2048_25: correct, 7.16 ms, 29.3 GB/s (effective, 2 B/site/sweep) (14.6% of 200 GB/s)
  score (gmean of fraction): 0.0982

## History

- iter  2: compile=OK | correct=True | score=0.06639990940563656
- iter  3: compile=OK | correct=True | score=0.05105530553349914
- iter  4: compile=OK | correct=True | score=0.04320344429157606
- iter  5: compile=OK | correct=True | score=0.06540463714287686
- iter  6: compile=OK | correct=True | score=0.07905155275612645
- iter  7: compile=OK | correct=True | score=0.09821411225912506
- iter  8: compile=OK | correct=True | score=0.05806948484121487
- iter  9: compile=OK | correct=True | score=0.050997447759371144

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
