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

#define TGW 16
#define TGH 16

kernel void ising_step(device       char  *spins    [[buffer(0)]],
                       device const float *p_accept [[buffer(1)]],
                       constant uint  &NX           [[buffer(2)]],
                       constant uint  &NY           [[buffer(3)]],
                       constant uint  &step_idx     [[buffer(4)]],
                       constant uint  &seed         [[buffer(5)]],
                       uint2 gid  [[thread_position_in_grid]],
                       uint2 lid2 [[thread_position_in_threadgroup]],
                       uint  lid  [[thread_index_in_threadgroup]],
                       uint2 tgid [[threadgroup_position_in_grid]])
{
    threadgroup float pa_tab[5];
    threadgroup char  tile[(TGH + 2) * (TGW + 2)];

    // Cache p_accept (5 entries).
    if (lid < 5u) {
        pa_tab[lid] = p_accept[lid];
    }

    // Origin of the tile in lattice coords (top-left of inner region).
    int base_i = (int)(tgid.x * TGW);
    int base_j = (int)(tgid.y * TGH);
    int nx = (int)NX;
    int ny = (int)NY;

    // Cooperatively load (TGH+2) x (TGW+2) tile including halo.
    // Total = 324 elements, threadgroup has 256 threads => up to 2 per thread.
    const uint TILE_W = TGW + 2;
    const uint TILE_H = TGH + 2;
    const uint TILE_N = TILE_W * TILE_H; // 324
    const uint TG_N   = TGW * TGH;       // 256

    for (uint t = lid; t < TILE_N; t += TG_N) {
        uint ty = t / TILE_W;
        uint tx = t - ty * TILE_W;
        int gx = base_i + (int)tx - 1;
        int gy = base_j + (int)ty - 1;
        // Periodic wrap.
        if (gx < 0)      gx += nx;
        else if (gx >= nx) gx -= nx;
        if (gy < 0)      gy += ny;
        else if (gy >= ny) gy -= ny;
        tile[t] = spins[gy * nx + gx];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint i = gid.x;
    uint j = gid.y;
    if (i >= NX || j >= NY) return;

    uint color = step_idx & 1u;
    bool active = (((i + j) & 1u) == color);

    // Local tile coordinates of this site (offset by +1 for halo).
    uint tx = lid2.x + 1u;
    uint ty = lid2.y + 1u;

    int s  = (int)tile[ty * TILE_W + tx];
    int sl = (int)tile[ty * TILE_W + (tx - 1u)];
    int sr = (int)tile[ty * TILE_W + (tx + 1u)];
    int sd = (int)tile[(ty - 1u) * TILE_W + tx];
    int su = (int)tile[(ty + 1u) * TILE_W + tx];

    int h    = sl + sr + sd + su;
    int prod = s * h;
    int idx  = (prod + 4) >> 1;
    float pa = pa_tab[idx];

    uint site_idx = j * NX + i;
    uint x = seed + step_idx * 0x9E3779B9u;
    x = mix32(x);
    x = mix32(x ^ site_idx);
    float u = float(x >> 8) * (1.0f / 16777216.0f);

    if (active && (u < pa)) {
        spins[site_idx] = (char)(-s);
    }
}
```

Result of previous attempt:
       256x256_100: correct, 3.63 ms, 3.6 GB/s (effective, 2 B/site/sweep) (1.8% of 200 GB/s)
      1024x1024_50: correct, 6.47 ms, 16.2 GB/s (effective, 2 B/site/sweep) (8.1% of 200 GB/s)
      2048x2048_25: correct, 11.98 ms, 17.5 GB/s (effective, 2 B/site/sweep) (8.8% of 200 GB/s)
  score (gmean of fraction): 0.0504

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

- iter  0: compile=OK | correct=True | score=0.06647490404968624
- iter  1: compile=OK | correct=True | score=0.075244898461611
- iter  2: compile=FAIL | correct=False | score=N/A
- iter  3: compile=OK | correct=False | score=N/A
- iter  4: compile=OK | correct=True | score=0.06649708196274966
- iter  5: compile=OK | correct=True | score=0.05041289677691749

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
