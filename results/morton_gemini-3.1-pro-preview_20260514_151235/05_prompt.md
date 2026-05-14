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

    constexpr uint X_MASK  = 0x09249249u;
    constexpr uint Y_MASK  = 0x12492492u;
    constexpr uint Z_MASK  = 0x24924924u;
    
    constexpr uint YZ_MASK = Y_MASK | Z_MASK;
    constexpr uint XZ_MASK = X_MASK | Z_MASK;
    constexpr uint XY_MASK = X_MASK | Y_MASK;

    uint m = tid;
    float c = u_in[m];

    uint valid_mask = (1u << (3u * logN)) - 1u;
    uint x_mask = X_MASK & valid_mask;
    uint y_mask = Y_MASK & valid_mask;
    uint z_mask = Z_MASK & valid_mask;

    // Fast global boundary check
    uint not_m = ~m;
    bool is_boundary = ((m & x_mask) == 0u) | ((not_m & x_mask) == 0u) |
                       ((m & y_mask) == 0u) | ((not_m & y_mask) == 0u) |
                       ((m & z_mask) == 0u) | ((not_m & z_mask) == 0u);

    // Identify local 4x4x2 block coordinates within the SIMD group
    uint lane = m & 31u;
    uint lx = (lane & 1u) | ((lane >> 2u) & 2u);
    uint ly = ((lane >> 1u) & 1u) | ((lane >> 3u) & 2u);
    uint lz = (lane >> 2u) & 1u;

    // Calculate shuffle lanes for intra-block neighbors
    uint xp_lane = (lane + ((lx == 1u) ? 7u : 1u)) & 31u;
    uint xm_lane = (lane - ((lx == 2u) ? 7u : 1u)) & 31u;
    
    uint yp_lane = (lane + ((ly == 1u) ? 14u : 2u)) & 31u;
    uint ym_lane = (lane - ((ly == 2u) ? 14u : 2u)) & 31u;
    
    uint zp_lane = (lane + 4u) & 31u;
    uint zm_lane = (lane - 4u) & 31u;

    // Unconditionally execute shuffles for uniform SIMD behavior
    float xp = simd_shuffle(c, xp_lane);
    float xm = simd_shuffle(c, xm_lane);
    float yp = simd_shuffle(c, yp_lane);
    float ym = simd_shuffle(c, ym_lane);
    float zp = simd_shuffle(c, zp_lane);
    float zm = simd_shuffle(c, zm_lane);

    // Dirichlet BC guarantees output is unmodified on cube faces
    if (!is_boundary) {
        uint m_yz = m & YZ_MASK;
        uint m_xz = m & XZ_MASK;
        uint m_xy = m & XY_MASK;

        // Masking with valid_mask guarantees safe wraparound bounds if fetched
        if (lx == 3u) xp = u_in[((((m | YZ_MASK) + 1u) & X_MASK) | m_yz) & valid_mask];
        if (lx == 0u) xm = u_in[((((m & X_MASK) - 1u) & X_MASK) | m_yz) & valid_mask];
        
        if (ly == 3u) yp = u_in[((((m | XZ_MASK) + 2u) & Y_MASK) | m_xz) & valid_mask];
        if (ly == 0u) ym = u_in[((((m & Y_MASK) - 2u) & Y_MASK) | m_xz) & valid_mask];
        
        if (lz == 1u) zp = u_in[((((m | XY_MASK) + 4u) & Z_MASK) | m_xy) & valid_mask];
        if (lz == 0u) zm = u_in[((((m & Z_MASK) - 4u) & Z_MASK) | m_xy) & valid_mask];

        float sum = xm + xp + ym + yp + zm + zp;
        u_out[m] = c + alpha * (sum - 6.0f * c);
    } else {
        u_out[m] = c;
    }
}
```

Result of previous attempt:
           N32_120: correct, 2.80 ms, 11.2 GB/s (effective, 8 B/cell) (5.6% of 200 GB/s)
            N64_60: correct, 2.40 ms, 52.4 GB/s (effective, 8 B/cell) (26.2% of 200 GB/s)
           N128_30: correct, 8.54 ms, 58.9 GB/s (effective, 8 B/cell) (29.5% of 200 GB/s)
  score (gmean of fraction): 0.1631

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

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

    constexpr uint X_MASK  = 0x09249249u;
    constexpr uint Y_MASK  = 0x12492492u;
    constexpr uint Z_MASK  = 0x24924924u;
    
    constexpr uint YZ_MASK = Y_MASK | Z_MASK;
    constexpr uint XZ_MASK = X_MASK | Z_MASK;
    constexpr uint XY_MASK = X_MASK | Y_MASK;

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

    // Direct Morton neighbour index arithmetic: O(1) computation.
    // By saturating the stationary axes with 1s, an addition/subtraction
    // properly propagates carries exclusively along the desired axis bits.
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

Incumbent result:
           N32_120: correct, 1.36 ms, 23.1 GB/s (effective, 8 B/cell) (11.6% of 200 GB/s)
            N64_60: correct, 2.32 ms, 54.1 GB/s (effective, 8 B/cell) (27.1% of 200 GB/s)
           N128_30: correct, 3.34 ms, 150.5 GB/s (effective, 8 B/cell) (75.2% of 200 GB/s)
  score (gmean of fraction): 0.2867

## History

- iter  0: compile=OK | correct=True | score=0.07649270554108457
- iter  1: compile=FAIL | correct=False | score=N/A
- iter  2: compile=OK | correct=True | score=0.2867036352571831
- iter  3: compile=FAIL | correct=False | score=N/A
- iter  4: compile=OK | correct=True | score=0.16311843894973352

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
