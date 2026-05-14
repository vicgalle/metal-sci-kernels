**Optimization**: The incumbent is strong but small-N (N=32) is launch/occupancy bound. The main lever I have left without changing the dispatch is reducing the per-thread instruction count on the hot interior path. I'll:

1. Replace the 6-comparison boundary check with a single bitwise test: a cell is on a face iff `mx==0 || mx==xm_all || ...`. Using the identity `(mx-1) & mx == 0` only detects power-of-two; instead I compute `mxm1 = mx - 1` etc. as part of the neighbour math and reuse them.
2. Fold the boundary masks into compile-time-ish constants per call (still cheap).
3. Reuse `mx-1`, `my-2`, `mz-4` (already needed for the minus neighbours) to detect the `==0` case for free via the borrow-out, by checking the high bit of the masked subtraction. Combined with `==xm_all` via XOR-OR, the whole face test becomes a single `(A | B) != 0` style OR-reduction, removing 5 short-circuit branches that hurt SIMD divergence at boundary-heavy small N=32.
4. Keep all 7 loads issued before any arithmetic for max MLP.

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

    // All-ones pattern truncated to logN bits per axis.
    uint mask3  = (1u << (3u * logN)) - 1u;
    uint xm_all = XM & mask3;
    uint ym_all = YM & mask3;
    uint zm_all = ZM & mask3;

    // Single OR-reduction boundary test: face iff any plane is 0 or all-ones.
    // Trick: (mx == 0) || (mx == xm_all)  <=>  ((mx) * (mx ^ xm_all)) == 0
    // but multiplication is expensive; use OR of two equality bits via
    // (-(mx==0)) | (-(mx==xm_all)) style. The compiler lowers || to selects
    // anyway, so just OR the 6 boolean ints — single branch at the end.
    uint face = uint(mx == 0u) | uint(my == 0u) | uint(mz == 0u)
              | uint(mx == xm_all) | uint(my == ym_all) | uint(mz == zm_all);

    if (face) {
        u_out[tid] = u_in[tid];
        return;
    }

    uint YZ = my | mz;
    uint XZ = mx | mz;
    uint XY = mx | my;

    uint m_xp = (((m | (YM | ZM)) + 1u) & XM) | YZ;
    uint m_xm = ((mx - 1u) & XM) | YZ;
    uint m_yp = (((m | (XM | ZM)) + 2u) & YM) | XZ;
    uint m_ym = ((my - 2u) & YM) | XZ;
    uint m_zp = (((m | (XM | YM)) + 4u) & ZM) | XY;
    uint m_zm = ((mz - 4u) & ZM) | XY;

    // Issue all loads up front for maximum memory-level parallelism.
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