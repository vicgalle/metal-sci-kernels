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
    x = x ^ (x >> 16);
    return x;
}

#define TX 32u
#define TY 8u
#define TW (TX + 2u)
#define TH (TY + 2u)

kernel
[[max_total_threads_per_threadgroup(256)]]
void ising_step(device       char  *spins    [[buffer(0)]],
                device const float *p_accept [[buffer(1)]],
                constant uint  &NX           [[buffer(2)]],
                constant uint  &NY           [[buffer(3)]],
                constant uint  &step_idx     [[buffer(4)]],
                constant uint  &seed         [[buffer(5)]],
                uint2 gid  [[thread_position_in_grid]],
                uint2 lid  [[thread_position_in_threadgroup]],
                uint2 tgid [[threadgroup_position_in_grid]])
{
    threadgroup float pa_tab[5];
    threadgroup char  tile[TH * TW];

    uint lx = lid.x;
    uint ly = lid.y;
    uint flat = ly * TX + lx;
    const uint total = TX * TY; // 256

    if (flat < 5u) {
        pa_tab[flat] = p_accept[flat];
    }

    uint i0 = tgid.x * TX;
    uint j0 = tgid.y * TY;

    // Cooperative halo load: TH * TW = 10 * 34 = 340 cells, 256 threads.
    const uint tile_n = TH * TW;
    for (uint k = flat; k < tile_n; k += total) {
        uint ty_ = k / TW;
        uint tx_ = k - ty_ * TW;

        // Global lattice coords with periodic wrap. Use int math to avoid uint underflow.
        int gi_s = (int)i0 + (int)tx_ - 1;
        int gj_s = (int)j0 + (int)ty_ - 1;
        if (gi_s < 0)        gi_s += (int)NX;
        if (gi_s >= (int)NX) gi_s -= (int)NX;
        if (gj_s < 0)        gj_s += (int)NY;
        if (gj_s >= (int)NY) gj_s -= (int)NY;

        tile[ty_ * TW + tx_] = spins[(uint)gj_s * NX + (uint)gi_s];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint i = gid.x;
    uint j = gid.y;
    if (i >= NX || j >= NY) return;

    uint color = step_idx & 1u;
    if (((i + j) & 1u) != color) return;

    uint tx = lx + 1u;
    uint ty = ly + 1u;

    int s  = (int)tile[ty * TW + tx];
    int sl = (int)tile[ty * TW + (tx - 1u)];
    int sr = (int)tile[ty * TW + (tx + 1u)];
    int sd = (int)tile[(ty - 1u) * TW + tx];
    int su = (int)tile[(ty + 1u) * TW + tx];

    int h    = sl + sr + sd + su;
    int prod = s * h;
    int idx  = (prod + 4) >> 1;
    float pa = pa_tab[idx];

    uint site_idx = j * NX + i;
    uint x = seed + step_idx * 0x9E3779B9u;
    x = mix32(x);
    x = mix32(x ^ site_idx);
    float u = float(x >> 8) * (1.0f / 16777216.0f);

    char new_s = (char)(-s);
    char old_s = (char)s;
    spins[site_idx] = (u < pa) ? new_s : old_s;
}
```

Result of previous attempt:
  COMPILE FAILED: Error Domain=MTLLibraryErrorDomain Code=3 "program_source:17:1: error: an attribute list cannot appear here
[[max_total_threads_per_threadgroup(256)]]
^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
" UserInfo={NSLocalizedDescription=program_source:17:1: error: an attribute list cannot appear here
[[max_total_threads_per_threadgroup(256)]]
^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

inline uint mix32(uint x) {
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    x = x ^ (x >> 16);
    return x;
}

kernel void ising_step(device       char  *spins    [[buffer(0)]],
                       device const float *p_accept [[buffer(1)]],
                       constant uint  &NX           [[buffer(2)]],
                       constant uint  &NY           [[buffer(3)]],
                       constant uint  &step_idx     [[buffer(4)]],
                       constant uint  &seed         [[buffer(5)]],
                       uint2 gid [[thread_position_in_grid]],
                       uint   lid [[thread_index_in_threadgroup]]) {
    threadgroup float pa_tab[5];
    if (lid < 5u) {
        pa_tab[lid] = p_accept[lid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint i = gid.x;
    uint j = gid.y;
    if (i >= NX || j >= NY) return;

    uint color = step_idx & 1u;
    if (((i + j) & 1u) != color) return;

    // Branchless periodic neighbours.
    uint im = select(i - 1u, NX - 1u, i == 0u);
    uint ip = select(i + 1u, 0u,      i == NX - 1u);
    uint jm = select(j - 1u, NY - 1u, j == 0u);
    uint jp = select(j + 1u, 0u,      j == NY - 1u);

    uint row   = j  * NX;
    uint row_m = jm * NX;
    uint row_p = jp * NX;

    int s  = (int)spins[row   + i ];
    int sl = (int)spins[row   + im];
    int sr = (int)spins[row   + ip];
    int sd = (int)spins[row_m + i ];
    int su = (int)spins[row_p + i ];

    int h    = sl + sr + sd + su;
    int prod = s * h;
    int idx  = (prod + 4) >> 1;
    float pa = pa_tab[idx];

    uint site_idx = row + i;
    uint x = seed + step_idx * 0x9E3779B9u;
    x = mix32(x);
    x = mix32(x ^ site_idx);
    float u = float(x >> 8) * (1.0f / 16777216.0f);

    char new_s = (char)(-s);
    char old_s = (char)s;
    spins[site_idx] = (u < pa) ? new_s : old_s;
}
```

Incumbent result:
       256x256_100: correct, 1.84 ms, 7.1 GB/s (effective, 2 B/site/sweep) (3.6% of 200 GB/s)
      1024x1024_50: correct, 5.06 ms, 20.7 GB/s (effective, 2 B/site/sweep) (10.4% of 200 GB/s)
      2048x2048_25: correct, 9.08 ms, 23.1 GB/s (effective, 2 B/site/sweep) (11.5% of 200 GB/s)
  score (gmean of fraction): 0.0752

## History

- iter  2: compile=FAIL | correct=False | score=N/A
- iter  3: compile=OK | correct=False | score=N/A
- iter  4: compile=OK | correct=True | score=0.06649708196274966
- iter  5: compile=OK | correct=True | score=0.05041289677691749
- iter  6: compile=OK | correct=True | score=0.06522928648163716
- iter  7: compile=FAIL | correct=False | score=N/A
- iter  8: compile=OK | correct=False | score=N/A
- iter  9: compile=FAIL | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
