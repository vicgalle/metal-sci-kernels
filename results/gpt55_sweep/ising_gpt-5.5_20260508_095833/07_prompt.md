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

Result of previous attempt:
       256x256_100: correct, 2.89 ms, 4.5 GB/s (effective, 2 B/site/sweep) (2.3% of 200 GB/s)
      1024x1024_50: correct, 3.91 ms, 26.8 GB/s (effective, 2 B/site/sweep) (13.4% of 200 GB/s)
      2048x2048_25: correct, 6.46 ms, 32.5 GB/s (effective, 2 B/site/sweep) (16.2% of 200 GB/s)
  score (gmean of fraction): 0.0791

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
- iter  4: compile=OK | correct=True | score=0.04320344429157606
- iter  5: compile=OK | correct=True | score=0.06540463714287686
- iter  6: compile=OK | correct=True | score=0.07905155275612645

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
