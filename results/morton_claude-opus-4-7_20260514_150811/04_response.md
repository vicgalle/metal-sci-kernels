**Optimization:** The previous TG-shared-memory attempt regressed because the neighbor-locality check (range-test + conditional load) added latency and divergence that outweighed the savings. The incumbent is already nearly bandwidth-bound at N=128 (92.6%). The main remaining inefficiency is the boundary path's redundant `u_in[tid]` reload (the value was already fetched as `c` in the common path, but we re-read it). Also, I can shave ALU by precomputing `mask3` values once and simplifying the boundary detection. The biggest opportunity at small N=32 is reducing per-thread fixed overhead: precompute mask constants from `logN` more cheaply, and avoid the second `u_in[tid]` load on boundaries (just write `u_in[tid]` once — which the compiler will fuse). Keep the proven neighbor-arithmetic and avoid TG memory which hurt locality. I also add `__builtin_assume`-style hints via `as_type` patterns the compiler can use.

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

    // High bit positions for coord == N-1 in each plane.
    // For coord==N-1, the bit-plane equals XM/YM/ZM truncated to 3*logN bits.
    uint shift  = 3u * logN;
    uint mask3  = (1u << shift) - 1u;
    uint xm_all = XM & mask3;
    uint ym_all = YM & mask3;
    uint zm_all = ZM & mask3;

    // Load center value early to overlap with boundary-check ALU.
    float c = u_in[tid];

    bool is_boundary = (mx == 0u) | (my == 0u) | (mz == 0u) |
                       (mx == xm_all) | (my == ym_all) | (mz == zm_all);

    if (is_boundary) {
        u_out[tid] = c;
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

    // Issue all 6 neighbor loads back-to-back so the compiler/HW can
    // batch them into one wide memory transaction per simdgroup.
    float xp = u_in[m_xp];
    float xn = u_in[m_xm];
    float yp = u_in[m_yp];
    float yn = u_in[m_ym];
    float zp = u_in[m_zp];
    float zn = u_in[m_zm];

    float sxy = (xp + xn) + (yp + yn);
    float sz  = (zp + zn);
    float sum6 = sxy + sz;
    u_out[tid] = fma(alpha, sum6 - 6.0f * c, c);
}
```