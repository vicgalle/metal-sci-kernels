## Task: morton

3D heat-equation stencil with Morton (Z-order) buffer layout. For each cell at (x, y, z) ∈ [0, N)^3, the linear buffer index is the bit-interleave
  M(x, y, z) = sum_{i=0}^{logN-1} ( x[i]·2^(3i)
                                    + y[i]·2^(3i+1)
                                    + z[i]·2^(3i+2) )
with x[i] the i-th bit of x and logN = log2(N). N is a power of 2 in every test (both in-distribution and held-out), so M is a bijection onto [0, N^3) and the buffer is exactly N^3 floats with no padding.

One forward-Euler timestep, 7-point Laplacian:
  u_new[M(x,y,z)] = u[M(x,y,z)] + alpha * (
         u[M(x-1,y,z)] + u[M(x+1,y,z)]
       + u[M(x,y-1,z)] + u[M(x,y+1,z)]
       + u[M(x,y,z-1)] + u[M(x,y,z+1)]
       - 6 u[M(x,y,z)] )
Stability requires alpha < 1/6 for the 3D 7-point heat stencil; the host uses alpha = 0.10. Dirichlet BC: every cell with x, y, or z in {0, N-1} (a face of the cube) MUST copy u → u_new unchanged. The initial state has those faces hard-zero. The host ping-pongs u_in/u_out across n_steps timesteps in one command buffer for accurate end-to-end GPU timing.

Optimization lever, unique to this task:
  (a) Morton encode/decode efficiency. The seed uses an O(logN) per-bit loop. Magic-constant bit spreading (PDEP-style) is O(1):
    uint spread3(uint v) {  // pack 8 bits at stride 3
        v = (v | (v << 16)) & 0x030000FFu;
        v = (v | (v <<  8)) & 0x0300F00Fu;
        v = (v | (v <<  4)) & 0x030C30C3u;
        v = (v | (v <<  2)) & 0x09249249u;
        return v;
    }
    uint M(uint x, uint y, uint z) {
        return spread3(x) | (spread3(y)<<1) | (spread3(z)<<2);
    }
  256-entry per-byte lookup tables in constant memory are an alternative that trades register pressure for arithmetic.
  (b) Neighbour-index arithmetic on the Morton index directly, avoiding the encode round-trip. With
    X_MASK = 0x09249249u  // bits 0, 3, 6, ... up to 3·logN
    Y_MASK = 0x12492492u  // bits 1, 4, 7, ...
    Z_MASK = 0x24924924u  // bits 2, 5, 8, ...
  (each truncated to 3·logN bits) one has
    m_xp = ((m | (Y_MASK | Z_MASK)) + 1u) & X_MASK
             | (m & (Y_MASK | Z_MASK));
    m_xm = ((m & X_MASK) - 1u) & X_MASK
             | (m & (Y_MASK | Z_MASK));
  and analogous formulas for ± y, ± z by rotating the masks.
  (c) Cache locality of the Morton traversal. Consecutive Morton indices cluster spatially: for logN ≥ 3 a 32-thread simdgroup covers a 4·2·4 block, so its 6 stencil neighbours reuse L1/SLC heavily. Threads MUST be dispatched 1-D with tid = Morton index — that is what the seed does and what the locality argument requires.

The in-distribution sizes (32, 64, 128) are SLC-resident on M1 Pro; the held-out 256^3 has a 128 MB ping-pong working set and is solidly DRAM-bound, so the cache-locality lever actually matters there. A candidate that wins in-distribution by pure encode/decode speedups without delivering locality will reveal that at the held-out size.

## Required kernel signature(s)

```
kernel void morton_stencil(
    device const float *u_in   [[buffer(0)]],
    device       float *u_out  [[buffer(1)]],
    constant uint      &N      [[buffer(2)]],
    constant uint      &logN   [[buffer(3)]],
    constant float     &alpha  [[buffer(4)]],
    uint tid [[thread_position_in_grid]]);

Dispatch geometry (host-fixed): 1-D dispatch of N^3 threads padded up to a multiple of the chosen TG width; the host picks tg_width = min(256, maxTotalThreadsPerThreadgroup). Threads MUST early-exit if tid >= N^3.

Convention: tid is the MORTON INDEX (not a (x,y,z) linear position). Consecutive threads therefore access consecutive buffer elements u_in[tid] / u_out[tid] — this is the trait Morton ordering exists to exploit; the kernel must keep it. Inside the kernel: decode tid → (x, y, z) for the boundary check, then compute the Morton indices of the six neighbours to gather the stencil. logN is provided as a separate constant so the kernel can iterate exactly logN times (5/6/7/8 across the size set) without a runtime log2; the host guarantees N == 1 << logN.

If you cap the kernel with [[max_total_threads_per_threadgroup(W)]], place the attribute on the kernel declaration itself; the host picks tg_width = min(W, 256). Buffers 0 and 1 are read/write; the host ping-pongs their roles across timesteps, so do NOT assume u_in and u_out alias fixed addresses.
```

## Your previous attempt

```metal
#include <metal_stdlib>
using namespace metal;

constexpr uint X_MASK = 0x09249249u;
constexpr uint Y_MASK = 0x12492492u;
constexpr uint Z_MASK = 0x24924924u;

constexpr uint YZ_MASK = Y_MASK | Z_MASK;
constexpr uint XZ_MASK = X_MASK | Z_MASK;
constexpr uint XY_MASK = X_MASK | Y_MASK;

[[max_total_threads_per_threadgroup(256)]]
kernel void morton_stencil(
    device const float *u_in   [[buffer(0)]],
    device       float *u_out  [[buffer(1)]],
    constant uint      &N      [[buffer(2)]],
    constant uint      &logN   [[buffer(3)]],
    constant float     &alpha  [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = N * N * N;
    if (tid >= total) return;

    uint m = tid;
    float c = u_in[m];

    // Compute exactly which bits are valid for the current grid size 
    uint valid_mask = (1u << (3u * logN)) - 1u;
    
    uint x_mask = X_MASK & valid_mask;
    uint y_mask = Y_MASK & valid_mask;
    uint z_mask = Z_MASK & valid_mask;

    uint m_x = m & x_mask;
    uint m_y = m & y_mask;
    uint m_z = m & z_mask;

    // Dirichlet BC: boundaries mean a dimension is either at 0 (all respective bits 0) 
    // or at N - 1 (all respective valid bits 1).
    if (m_x == 0u || m_x == x_mask ||
        m_y == 0u || m_y == y_mask ||
        m_z == 0u || m_z == z_mask) {
        u_out[m] = c;
        return;
    }

    uint m_yz = m & YZ_MASK;
    uint m_xz = m & XZ_MASK;
    uint m_xy = m & XY_MASK;

    // Direct Morton neighbour index arithmetic
    uint m_xp = (((m | YZ_MASK) + 1u) & X_MASK) | m_yz;
    uint m_xm = (((m & X_MASK) - 1u) & X_MASK) | m_yz;

    uint m_yp = (((m | XZ_MASK) + 2u) & Y_MASK) | m_xz;
    uint m_ym = (((m & Y_MASK) - 2u) & Y_MASK) | m_xz;

    uint m_zp = (((m | XY_MASK) + 4u) & Z_MASK) | m_xy;
    uint m_zm = (((m & Z_MASK) - 4u) & Z_MASK) | m_xy;

    float xm = u_in[m_xm];
    float xp = u_in[m_xp];
    float ym = u_in[m_ym];
    float yp = u_in[m_yp];
    float zm = u_in[m_zm];
    float zp = u_in[m_zp];

    u_out[m] = c + alpha * (xm + xp + ym + yp + zm + zp - 6.0f * c);
}
```

Result of previous attempt:
  COMPILE FAILED: Error Domain=MTLLibraryErrorDomain Code=3 "program_source:4:16: error: program scope variable must reside in constant address space
constexpr uint X_MASK = 0x09249249u;
               ^
program_source:5:16: error: program scope variable must reside in constant address space
constexpr uint Y_MASK = 0x12492492u;
               ^
program_source:6:16: error: program scope variable must reside in constant address space
constexpr uint Z_MASK = 0x24924924u;
               ^
program_source:8:16: error: program scope variable must reside in constant address space
constexpr uint YZ_MASK = Y_MASK | Z_MASK;
               ^
program_source:9:16: error: program scope variable must reside in constant address space
constexpr uint XZ_MASK = X_MASK | Z_MASK;
               ^
program_source:10:16: error: program scope variable must reside in constant address space
constexpr uint XY_MASK = X_MASK | Y_MASK;
               ^
" UserInfo={NSLocalizedDescription=program_source:4:16: error: program scope variable must reside in constant address space
constexpr uint X_MASK = 0x09249249u;
               ^
program_source:5:16: error: program scope variable must reside in constant address space
constexpr uint Y_MASK = 0x12492492u;
               ^
program_source:6:16: error: program scope variable must reside in constant address space
constexpr uint Z_MASK = 0x24924924u;
               ^
program_source:8:16: error: program scope variable must reside in constant address space
constexpr uint YZ_MASK = Y_MASK | Z_MASK;
               ^
program_source:9:16: error: program scope variable must reside in constant address space
constexpr uint XZ_MASK = X_MASK | Z_MASK;
               ^
program_source:10:16: error: program scope variable must reside in constant address space
constexpr uint XY_MASK = X_MASK | Y_MASK;
               ^
}

## Current best (incumbent)

```metal
// Naive seed for a Morton-ordered 3D heat-equation stencil.
//
// State is a 1D float buffer of length N^3 (N a power of 2), indexed in
// Z-order (Morton) layout: the value at coordinates (x, y, z) ∈ [0, N)^3
// lives at linear index
//
//     M(x, y, z) = sum_{i=0}^{logN-1} ( x[i]·2^(3i)
//                                       + y[i]·2^(3i+1)
//                                       + z[i]·2^(3i+2) )
//
// where x[i] is the i-th bit of x and logN = log2(N). M is a bijection
// onto [0, N^3) so the buffer is exactly N^3 floats with no padding.
//
// One timestep, forward Euler, 7-point Laplacian:
//
//   u_new[M(x,y,z)] = u[M(x,y,z)] + alpha * ( u[M(x-1,y,z)] + u[M(x+1,y,z)]
//                                           + u[M(x,y-1,z)] + u[M(x,y+1,z)]
//                                           + u[M(x,y,z-1)] + u[M(x,y,z+1)]
//                                           - 6 u[M(x,y,z)] )
//
// Dirichlet boundary: a cell with x, y, or z in {0, N-1} (i.e. on a face
// of the cube) copies u → u_new unchanged. The host ping-pongs u_in and
// u_out across n_steps timesteps in one command buffer.
//
// Buffer layout:
//   buffer 0: const float* u_in     (N^3, Morton-ordered)
//   buffer 1: device float* u_out   (N^3, Morton-ordered)
//   buffer 2: const uint& N         (grid size, power of 2)
//   buffer 3: const uint& logN      (= log2(N); host guarantees N == 1<<logN)
//   buffer 4: const float& alpha    (alpha = dt / dx^2 ∈ [0, 1/6])
//
// Convention: tid is the MORTON INDEX. Decode tid → (x, y, z) for the
// boundary check, then compute the Morton indices of the six neighbours
// to gather the stencil. The candidate is expected to find a faster
// Morton encode/decode (magic-constant bit-spread or lookup table) and/or
// neighbour-index bit twiddling that avoids the full encode round-trip.

#include <metal_stdlib>
using namespace metal;

inline uint morton_encode_3d(uint x, uint y, uint z, uint logN) {
    uint m = 0u;
    for (uint i = 0u; i < logN; ++i) {
        m |= ((x >> i) & 1u) << (3u * i + 0u);
        m |= ((y >> i) & 1u) << (3u * i + 1u);
        m |= ((z >> i) & 1u) << (3u * i + 2u);
    }
    return m;
}

inline void morton_decode_3d(uint m, uint logN,
                             thread uint &x, thread uint &y, thread uint &z) {
    x = 0u; y = 0u; z = 0u;
    for (uint i = 0u; i < logN; ++i) {
        x |= ((m >> (3u * i + 0u)) & 1u) << i;
        y |= ((m >> (3u * i + 1u)) & 1u) << i;
        z |= ((m >> (3u * i + 2u)) & 1u) << i;
    }
}

kernel void morton_stencil(device const float *u_in   [[buffer(0)]],
                           device       float *u_out  [[buffer(1)]],
                           constant uint      &N      [[buffer(2)]],
                           constant uint      &logN   [[buffer(3)]],
                           constant float     &alpha  [[buffer(4)]],
                           uint tid [[thread_position_in_grid]])
{
    uint total = N * N * N;
    if (tid >= total) return;

    uint x, y, z;
    morton_decode_3d(tid, logN, x, y, z);

    // Dirichlet BC: faces of the cube copy through unchanged.
    if (x == 0u || y == 0u || z == 0u ||
        x == N - 1u || y == N - 1u || z == N - 1u) {
        u_out[tid] = u_in[tid];
        return;
    }

    uint m_xm = morton_encode_3d(x - 1u, y,      z,      logN);
    uint m_xp = morton_encode_3d(x + 1u, y,      z,      logN);
    uint m_ym = morton_encode_3d(x,      y - 1u, z,      logN);
    uint m_yp = morton_encode_3d(x,      y + 1u, z,      logN);
    uint m_zm = morton_encode_3d(x,      y,      z - 1u, logN);
    uint m_zp = morton_encode_3d(x,      y,      z + 1u, logN);

    float c  = u_in[tid];
    float xm = u_in[m_xm];
    float xp = u_in[m_xp];
    float ym = u_in[m_ym];
    float yp = u_in[m_yp];
    float zm = u_in[m_zm];
    float zp = u_in[m_zp];

    u_out[tid] = c + alpha * (xm + xp + ym + yp + zm + zp - 6.0f * c);
}
```

Incumbent result:
           N32_120: correct, 2.49 ms, 12.7 GB/s (effective, 8 B/cell) (6.3% of 200 GB/s)
            N64_60: correct, 7.12 ms, 17.7 GB/s (effective, 8 B/cell) (8.8% of 200 GB/s)
           N128_30: correct, 31.43 ms, 16.0 GB/s (effective, 8 B/cell) (8.0% of 200 GB/s)
  score (gmean of fraction): 0.0765

## History

- iter  0: compile=OK | correct=True | score=0.07649270554108457
- iter  1: compile=FAIL | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
