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

Result of previous attempt:
       256x256_100: correct, 1.99 ms, 6.6 GB/s (effective, 2 B/site/sweep) (3.3% of 200 GB/s)
      1024x1024_50: correct, 8.56 ms, 12.2 GB/s (effective, 2 B/site/sweep) (6.1% of 200 GB/s)
      2048x2048_25: correct, 15.92 ms, 13.2 GB/s (effective, 2 B/site/sweep) (6.6% of 200 GB/s)
  score (gmean of fraction): 0.0511

## Current best (incumbent)

```metal
// Naive seed kernel for the 2D Ising model with checkerboard Metropolis
// updates. Spins are int8 +/-1 on a periodic NX x NY lattice.
//
// One sub-pass updates one color of the checkerboard:
//   color = step_idx & 1u   (0 = "red" where (i + j) is even,
//                            1 = "black" where (i + j) is odd)
// The host dispatches this kernel 2 * n_sweeps times with step_idx =
// 0, 1, 2, ... so each full sweep is one red pass + one black pass.
// All updates within a sub-pass are independent (neighbours of color c
// are color 1-c, untouched in this dispatch).
//
// Acceptance: for site (i, j) with current spin s and neighbour sum
// h = sl + sr + sd + su (h in {-4, -2, 0, 2, 4}), the energy change of
// flipping s is dE = 2 J s h with J = 1, so s*h in {-4, -2, 0, 2, 4}
// gives dE in {-8, -4, 0, 4, 8}. We pre-tabulate
//   p_accept[5] = {1, 1, 1, exp(-4 beta), exp(-8 beta)}
// indexed by (s*h + 4) / 2 in {0..4}. The seed reads exp(-...) values
// from the buffer rather than calling exp() so CPU reference and GPU
// kernel see bit-identical acceptance probabilities.
//
// RNG: a Murmur3-fmix32-style hash of (seed, step_idx, site_idx).
//   uint x = seed + step_idx * 0x9E3779B9u
//   x = mix(x);            // round 1
//   x = mix(x ^ site_idx); // round 2
//   u = float(x >> 8) * (1.0f / 16777216.0f);   // 24-bit uniform [0,1)
// The same hash is mirrored bit-for-bit by the host's numpy reference.
//
// Buffer layout (must be preserved by candidate kernels):
//   buffer 0: device char  *spins      (NX * NY, row-major, values +/-1)
//   buffer 1: device const float *p_accept   (5 entries; see above)
//   buffer 2: const uint  &NX
//   buffer 3: const uint  &NY
//   buffer 4: const uint  &step_idx    (host increments by 1 per dispatch)
//   buffer 5: const uint  &seed        (32-bit RNG seed)

#include <metal_stdlib>
using namespace metal;

inline uint mix32(uint x) {
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    x = x ^ (x >> 16);
    return x;
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
    uint i = gid.x;
    uint j = gid.y;
    if (i >= NX || j >= NY) return;

    uint color = step_idx & 1u;
    if (((i + j) & 1u) != color) return;   // wrong color sub-pass: skip

    // Periodic neighbours.
    uint im = (i == 0u)        ? (NX - 1u) : (i - 1u);
    uint ip = (i == NX - 1u)   ? 0u        : (i + 1u);
    uint jm = (j == 0u)        ? (NY - 1u) : (j - 1u);
    uint jp = (j == NY - 1u)   ? 0u        : (j + 1u);

    int s  = (int)spins[j  * NX + i ];
    int sl = (int)spins[j  * NX + im];
    int sr = (int)spins[j  * NX + ip];
    int sd = (int)spins[jm * NX + i ];
    int su = (int)spins[jp * NX + i ];

    int h    = sl + sr + sd + su;          // in {-4,-2,0,2,4}
    int prod = s * h;                      // in {-4,-2,0,2,4}
    int idx  = (prod + 4) / 2;             // in {0..4}
    float pa = p_accept[idx];

    uint site_idx = j * NX + i;
    uint bits = rand_u32(seed, step_idx, site_idx);
    float u = float(bits >> 8) * (1.0f / 16777216.0f);

    if (u < pa) {
        spins[j * NX + i] = (char)(-s);
    }
}
```

Incumbent result:
       256x256_100: correct, 1.17 ms, 11.2 GB/s (effective, 2 B/site/sweep) (5.6% of 200 GB/s)
      1024x1024_50: correct, 4.83 ms, 21.7 GB/s (effective, 2 B/site/sweep) (10.9% of 200 GB/s)
      2048x2048_25: correct, 8.69 ms, 24.1 GB/s (effective, 2 B/site/sweep) (12.1% of 200 GB/s)
  score (gmean of fraction): 0.0902

## History

- iter  0: compile=OK | correct=True | score=0.09018050267127434
- iter  1: compile=OK | correct=True | score=0.06422420489119225
- iter  2: compile=OK | correct=True | score=0.06639990940563656
- iter  3: compile=OK | correct=True | score=0.05105530553349914

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
