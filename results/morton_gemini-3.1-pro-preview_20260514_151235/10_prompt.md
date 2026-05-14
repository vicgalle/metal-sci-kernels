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
    uint total = 1u << (3u * logN);
    if (tid >= total) return;

    uint m = tid;
    
    // 1. Hoist center load and constants to hide latency behind integer ALU
    float c = u_in[m];
    float a = alpha;
    float c_weight = 1.0f - 6.0f * a;

    // 2. Vectorize Morton mask arithmetic into uint3 for optimal ILP
    constexpr uint3 MASK(0x49249249u, 0x92492492u, 0x24924924u);
    constexpr uint3 NOT_MASK = ~MASK;
    constexpr uint3 ADD_VAL(1u, 2u, 4u);

    // Scalar 'm' is implicitly promoted to uint3
    uint3 m_not = m & NOT_MASK;
    
    // O(1) Morton boundary-preserving arithmetic in parallel
    uint3 m_p = m_not | (((m | NOT_MASK) + ADD_VAL) & MASK);
    uint3 m_m = m_not | (((m & MASK) - ADD_VAL) & MASK);

    // 3. Consolidated out-of-bounds check catching all 6 boundaries
    uint3 m_bounds = m_m | m_p;
    if ((m_bounds.x | m_bounds.y | m_bounds.z) >= total) {
        u_out[m] = c;
        return;
    }

    // 4. Gather 6 neighbors (L1/SLC resident)
    float xm = u_in[m_m.x];
    float xp = u_in[m_p.x];
    float ym = u_in[m_m.y];
    float yp = u_in[m_p.y];
    float zm = u_in[m_m.z];
    float zp = u_in[m_p.z];

    // 5. Balanced accumulation tree
    float sum_x = xm + xp;
    float sum_y = ym + yp;
    float sum_z = zm + zp;
    float sum   = (sum_x + sum_y) + sum_z;

    u_out[m] = fma(sum, a, c * c_weight);
}
```

Result of previous attempt:
           N32_120: correct, 1.27 ms, 24.8 GB/s (effective, 8 B/cell) (12.4% of 200 GB/s)
            N64_60: correct, 1.90 ms, 66.1 GB/s (effective, 8 B/cell) (33.0% of 200 GB/s)
           N128_30: correct, 3.07 ms, 164.1 GB/s (effective, 8 B/cell) (82.0% of 200 GB/s)
  score (gmean of fraction): 0.3228

## History

- iter  2: compile=OK | correct=True | score=0.2867036352571831
- iter  3: compile=FAIL | correct=False | score=N/A
- iter  4: compile=OK | correct=True | score=0.16311843894973352
- iter  5: compile=OK | correct=True | score=0.2904362844232099
- iter  6: compile=OK | correct=True | score=0.15558260607106209
- iter  7: compile=OK | correct=True | score=0.3107642989753453
- iter  8: compile=OK | correct=True | score=0.1676186873462766
- iter  9: compile=OK | correct=True | score=0.32282201022366386

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
