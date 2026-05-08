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

Result of previous attempt:
       256x256_100: correct, 1.79 ms, 7.3 GB/s (effective, 2 B/site/sweep) (3.7% of 200 GB/s)
      1024x1024_50: correct, 7.44 ms, 14.1 GB/s (effective, 2 B/site/sweep) (7.0% of 200 GB/s)
      2048x2048_25: correct, 13.79 ms, 15.2 GB/s (effective, 2 B/site/sweep) (7.6% of 200 GB/s)
  score (gmean of fraction): 0.0581

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

- iter  1: compile=OK | correct=True | score=0.06422420489119225
- iter  2: compile=OK | correct=True | score=0.06639990940563656
- iter  3: compile=OK | correct=True | score=0.05105530553349914
- iter  4: compile=OK | correct=True | score=0.04320344429157606
- iter  5: compile=OK | correct=True | score=0.06540463714287686
- iter  6: compile=OK | correct=True | score=0.07905155275612645
- iter  7: compile=OK | correct=True | score=0.09821411225912506
- iter  8: compile=OK | correct=True | score=0.05806948484121487

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
