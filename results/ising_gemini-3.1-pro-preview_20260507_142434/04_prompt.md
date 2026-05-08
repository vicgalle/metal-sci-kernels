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

    // Checkerboard sub-pass mask
    uint color = step_idx & 1u;
    if (((i + j) & 1u) != color) return;

    // Precompute the site-invariant portion of the RNG hash.
    // The compiler can hoist this as a scalar uniform operation across the active warp.
    uint base_x = seed + step_idx * 0x9E3779B9u;
    base_x = (base_x ^ (base_x >> 16)) * 0x85EBCA6Bu;
    base_x = (base_x ^ (base_x >> 13)) * 0xC2B2AE35u;
    base_x = base_x ^ (base_x >> 16);

    // Periodic neighbor bounds calculation (branchless ternary)
    uint im = (i != 0u) ? (i - 1u) : (NX - 1u);
    uint ip = (i + 1u != NX) ? (i + 1u) : 0u;
    uint jm = (j != 0u) ? (j - 1u) : (NY - 1u);
    uint jp = (j + 1u != NY) ? (j + 1u) : 0u;

    // Pre-calculate base row pointers to eliminate repeated multiplications
    device char* row_curr = spins + j  * NX;
    device char* row_prev = spins + jm * NX;
    device char* row_next = spins + jp * NX;

    // L1 cache absorbs the unmasked byte-loads natively
    int s  = (int)row_curr[i];
    int sl = (int)row_curr[im];
    int sr = (int)row_curr[ip];
    int sd = (int)row_prev[i];
    int su = (int)row_next[i];

    int h    = sl + sr + sd + su;
    int prod = s * h;
    uint idx = (uint)(prod + 4) >> 1u;
    
    float pa = p_accept[idx];

    // Short-circuit: if acceptance probability is >= 1.0 (e.g. energy lowers), 
    // we bypass the costly RNG calculation entirely since u < 1.0 is always true.
    if (pa < 1.0f) {
        uint site_idx = j * NX + i;
        uint x = base_x ^ site_idx;
        
        // Second mix32 round
        x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
        x = (x ^ (x >> 13)) * 0xC2B2AE35u;
        x = x ^ (x >> 16);

        float u = float(x >> 8) * (1.0f / 16777216.0f);
        if (u >= pa) return;
    }

    // Flip accepted
    row_curr[i] = (char)(-s);
}
```

Result of previous attempt:
       256x256_100: correct, 3.70 ms, 3.5 GB/s (effective, 2 B/site/sweep) (1.8% of 200 GB/s)
      1024x1024_50: correct, 5.18 ms, 20.3 GB/s (effective, 2 B/site/sweep) (10.1% of 200 GB/s)
      2048x2048_25: correct, 9.37 ms, 22.4 GB/s (effective, 2 B/site/sweep) (11.2% of 200 GB/s)
  score (gmean of fraction): 0.0585

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
       256x256_100: correct, 1.20 ms, 10.9 GB/s (effective, 2 B/site/sweep) (5.4% of 200 GB/s)
      1024x1024_50: correct, 4.75 ms, 22.1 GB/s (effective, 2 B/site/sweep) (11.0% of 200 GB/s)
      2048x2048_25: correct, 8.66 ms, 24.2 GB/s (effective, 2 B/site/sweep) (12.1% of 200 GB/s)
  score (gmean of fraction): 0.0900

## History

- iter  0: compile=OK | correct=True | score=0.08995377405569595
- iter  1: compile=OK | correct=True | score=0.05482293274271089
- iter  2: compile=OK | correct=True | score=0.0331536084451446
- iter  3: compile=OK | correct=True | score=0.058532671495520674

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
