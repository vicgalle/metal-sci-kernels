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
kernel void morton_stencil(device const float *u_in   [[buffer(0)]],
                           device       float *u_out  [[buffer(1)]],
                           constant uint      &N      [[buffer(2)]],
                           constant uint      &logN   [[buffer(3)]],
                           constant float     &alpha  [[buffer(4)]],
                           uint tid [[thread_position_in_grid]])
{
    uint total = N * N * N;
    if (tid >= total) return;

    const uint XM = 0x09249249u;
    const uint YM = 0x12492492u;
    const uint ZM = 0x24924924u;

    uint m  = tid;
    uint mx = m & XM;
    uint my = m & YM;
    uint mz = m & ZM;

    uint mask3  = (1u << (3u * logN)) - 1u;
    uint xm_all = XM & mask3;
    uint ym_all = YM & mask3;
    uint zm_all = ZM & mask3;

    // Detect face membership per axis (branch-free).
    bool x0 = (mx == 0u);
    bool xN = (mx == xm_all);
    bool y0 = (my == 0u);
    bool yN = (my == ym_all);
    bool z0 = (mz == 0u);
    bool zN = (mz == zm_all);
    bool is_face = x0 | xN | y0 | yN | z0 | zN;

    uint YZ = my | mz;
    uint XZ = mx | mz;
    uint XY = mx | my;

    // Neighbour Morton indices via masked carry/borrow on each bit-plane.
    // For boundary cells we substitute tid so the gather returns u_in[tid]
    // on every direction, making sum6 == 6*c and u_out == c.
    uint m_xp = is_face ? tid : ((((m | (YM | ZM)) + 1u) & XM) | YZ);
    uint m_xm = is_face ? tid : (((mx - 1u) & XM) | YZ);
    uint m_yp = is_face ? tid : ((((m | (XM | ZM)) + 2u) & YM) | XZ);
    uint m_ym = is_face ? tid : (((my - 2u) & YM) | XZ);
    uint m_zp = is_face ? tid : ((((m | (XM | YM)) + 4u) & ZM) | XY);
    uint m_zm = is_face ? tid : (((mz - 4u) & ZM) | XY);

    float c  = u_in[tid];
    float xp = u_in[m_xp];
    float xn = u_in[m_xm];
    float yp = u_in[m_yp];
    float yn = u_in[m_ym];
    float zp = u_in[m_zp];
    float zn = u_in[m_zm];

    float sum6 = (xp + xn) + (yp + yn) + (zp + zn);
    u_out[tid] = fma(alpha, sum6 - 6.0f * c, c);
}
```

Result of previous attempt:
           N32_120: correct, 1.20 ms, 26.2 GB/s (effective, 8 B/cell) (13.1% of 200 GB/s)
            N64_60: correct, 2.12 ms, 59.2 GB/s (effective, 8 B/cell) (29.6% of 200 GB/s)
           N128_30: correct, 2.62 ms, 192.0 GB/s (effective, 8 B/cell) (96.0% of 200 GB/s)
  score (gmean of fraction): 0.3340

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(256)]]
kernel void morton_stencil(device const float *u_in   [[buffer(0)]],
                           device       float *u_out  [[buffer(1)]],
                           constant uint      &N      [[buffer(2)]],
                           constant uint      &logN   [[buffer(3)]],
                           constant float     &alpha  [[buffer(4)]],
                           uint tid [[thread_position_in_grid]])
{
    uint total = N * N * N;
    if (tid >= total) return;

    // Morton bit masks (full 32-bit; coords stay in [0,N) so high bits are 0).
    const uint XM = 0x09249249u;
    const uint YM = 0x12492492u;
    const uint ZM = 0x24924924u;

    uint m  = tid;
    uint mx = m & XM;
    uint my = m & YM;
    uint mz = m & ZM;

    // Boundary check without decoding: coord==0 iff its bit-plane is 0;
    // coord==N-1 iff its bit-plane equals the all-ones pattern truncated
    // to logN bits in that plane, i.e. XM & ((1u << (3*logN)) - 1u), etc.
    uint mask3 = (1u << (3u * logN)) - 1u;  // low 3*logN bits
    uint xm_all = XM & mask3;
    uint ym_all = YM & mask3;
    uint zm_all = ZM & mask3;

    if (mx == 0u || my == 0u || mz == 0u ||
        mx == xm_all || my == ym_all || mz == zm_all) {
        u_out[tid] = u_in[tid];
        return;
    }

    uint YZ = my | mz;
    uint XZ = mx | mz;
    uint XY = mx | my;

    // Neighbour Morton indices via masked carry/borrow on each bit-plane.
    uint m_xp = (((m | (YM | ZM)) + 1u) & XM) | YZ;
    uint m_xm = ((mx - 1u) & XM) | YZ;
    uint m_yp = (((m | (XM | ZM)) + 2u) & YM) | XZ;
    uint m_ym = ((my - 2u) & YM) | XZ;
    uint m_zp = (((m | (XM | YM)) + 4u) & ZM) | XY;
    uint m_zm = ((mz - 4u) & ZM) | XY;

    float c  = u_in[tid];
    float xp = u_in[m_xp];
    float xn = u_in[m_xm];
    float yp = u_in[m_yp];
    float yn = u_in[m_ym];
    float zp = u_in[m_zp];
    float zn = u_in[m_zm];

    float sum6 = (xp + xn) + (yp + yn) + (zp + zn);
    u_out[tid] = fma(alpha, sum6 - 6.0f * c, c);
}
```

Incumbent result:
           N32_120: correct, 0.60 ms, 52.4 GB/s (effective, 8 B/cell) (26.2% of 200 GB/s)
            N64_60: correct, 1.36 ms, 92.7 GB/s (effective, 8 B/cell) (46.4% of 200 GB/s)
           N128_30: correct, 2.72 ms, 185.1 GB/s (effective, 8 B/cell) (92.6% of 200 GB/s)
  score (gmean of fraction): 0.4825

## History

- iter  0: compile=OK | correct=True | score=0.07177634456919986
- iter  1: compile=OK | correct=True | score=0.3870279639559386
- iter  2: compile=OK | correct=True | score=0.48249521721992483
- iter  3: compile=OK | correct=True | score=0.29900674241656644
- iter  4: compile=OK | correct=True | score=0.2856219333410913
- iter  5: compile=OK | correct=True | score=0.2898003608806134
- iter  6: compile=OK | correct=True | score=0.33396441575005426

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
