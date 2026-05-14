**Optimization:** The incumbent already does direct neighbour-index arithmetic with no decode. The previous tiled attempt regressed because the in-tile check was expensive and the tile was tiny. I'll keep the clean incumbent structure but (1) hoist the boundary-mask constants out of per-thread work by using a simpler formulation, (2) eliminate the divergent early-return for boundary cells — instead, clamp neighbour reads using a select on the bit-plane being zero/full, which keeps all threads on the math path and lets the GPU coalesce memory traffic uniformly across the warp. For boundary cells we want `u_out = u_in`, which is achieved by making `sum6 == 6*c` (every neighbour reads back as `c`). This removes branch divergence at face cells and keeps the hot path branch-free, which should especially help N=32 where boundary cells are a large fraction.

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