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

kernel void morton_stencil(
    device const float *u_in   [[buffer(0)]],
    device       float *u_out  [[buffer(1)]],
    constant uint      &N      [[buffer(2)]],
    constant uint      &logN   [[buffer(3)]],
    constant float     &alpha  [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    (void)N;

    uint total = 1u << (3u * logN);
    if (tid >= total) return;

    constexpr uint X_FULL = 0x09249249u;
    constexpr uint Y_FULL = 0x12492492u;
    constexpr uint Z_FULL = 0x24924924u;

    uint validMask = total - 1u;
    uint xmask = X_FULL & validMask;
    uint ymask = Y_FULL & validMask;
    uint zmask = Z_FULL & validMask;

    uint mx = tid & xmask;
    uint my = tid & ymask;
    uint mz = tid & zmask;

    bool boundary = (mx == 0u) || (mx == xmask) ||
                    (my == 0u) || (my == ymask) ||
                    (mz == 0u) || (mz == zmask);

    float c = u_in[tid];

    uint m_xm = tid, m_xp = tid;
    uint m_ym = tid, m_yp = tid;
    uint m_zm = tid, m_zp = tid;

    if (!boundary) {
        uint yzmask = ymask | zmask;
        uint xzmask = xmask | zmask;
        uint xymask = xmask | ymask;

        m_xp = (((tid | yzmask) + 1u) & xmask) | (tid & yzmask);
        m_xm = ((mx - 1u) & xmask) | (tid & yzmask);

        m_yp = (((tid | xzmask) + 2u) & ymask) | (tid & xzmask);
        m_ym = ((my - 2u) & ymask) | (tid & xzmask);

        m_zp = (((tid | xymask) + 4u) & zmask) | (tid & xymask);
        m_zm = ((mz - 4u) & zmask) | (tid & xymask);
    }

    // Boundary lanes still participate so adjacent interior lanes can
    // safely shuffle their center values when the boundary neighbor is
    // inside the same SIMD group.
    float sxm = simd_shuffle(c, ushort(m_xm & 31u));
    float sxp = simd_shuffle(c, ushort(m_xp & 31u));
    float sym = simd_shuffle(c, ushort(m_ym & 31u));
    float syp = simd_shuffle(c, ushort(m_yp & 31u));
    float szm = simd_shuffle(c, ushort(m_zm & 31u));
    float szp = simd_shuffle(c, ushort(m_zp & 31u));

    if (boundary) {
        u_out[tid] = c;
        return;
    }

    constexpr uint SIMD_BASE_MASK = 0xffffffe0u;
    uint simdBase = tid & SIMD_BASE_MASK;

    float xm = sxm;
    if ((m_xm & SIMD_BASE_MASK) != simdBase) xm = u_in[m_xm];

    float xp = sxp;
    if ((m_xp & SIMD_BASE_MASK) != simdBase) xp = u_in[m_xp];

    float ym = sym;
    if ((m_ym & SIMD_BASE_MASK) != simdBase) ym = u_in[m_ym];

    float yp = syp;
    if ((m_yp & SIMD_BASE_MASK) != simdBase) yp = u_in[m_yp];

    float zm = szm;
    if ((m_zm & SIMD_BASE_MASK) != simdBase) zm = u_in[m_zm];

    float zp = szp;
    if ((m_zp & SIMD_BASE_MASK) != simdBase) zp = u_in[m_zp];

    u_out[tid] = c + alpha * (xm + xp + ym + yp + zm + zp - 6.0f * c);
}
```

Result of previous attempt:
           N32_120: correct, 1.36 ms, 23.2 GB/s (effective, 8 B/cell) (11.6% of 200 GB/s)
            N64_60: correct, 2.84 ms, 44.3 GB/s (effective, 8 B/cell) (22.2% of 200 GB/s)
           N128_30: correct, 8.07 ms, 62.3 GB/s (effective, 8 B/cell) (31.2% of 200 GB/s)
  score (gmean of fraction): 0.2000

## History

- iter  0: compile=OK | correct=True | score=0.07345154143890095
- iter  1: compile=OK | correct=True | score=0.20001246273698833

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
