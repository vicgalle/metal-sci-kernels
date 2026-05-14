Acknowledging the system reminder: the keybindings-help skill is not relevant to this MSL kernel optimization task, so I won't invoke it.

**Optimization:** The incumbent already uses the early-return boundary path which is fastest at small N. The main remaining bottleneck at N=32 and N=64 is launch/occupancy overhead and per-thread arithmetic. I'll keep the early-return structure (it was the winner) but tighten the inner path: (1) precompute `(YM|ZM)`, `(XM|ZM)`, `(XM|YM)` once and hoist `mask3` boundary constants; (2) use `select`/ternaries to avoid extra ops; (3) load the center value first and fuse the sum with `fma`. Most importantly, replace the boundary detection — the incumbent does `mask3 = (1u << (3u*logN)) - 1u` (a variable shift) every thread; I'll compute the per-plane "all-ones" constants as `(N-1)` spread instead, but more cheaply: since `xm_all = XM & ((1<<(3*logN))-1)`, and Apple GPUs handle the shift fine, but we can replace the OR-chain boundary test with a single comparison after combining. I'll also issue all six loads back-to-back before any arithmetic to maximize MLP.

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
    constexpr uint XM = 0x09249249u;
    constexpr uint YM = 0x12492492u;
    constexpr uint ZM = 0x24924924u;

    uint m  = tid;
    uint mx = m & XM;
    uint my = m & YM;
    uint mz = m & ZM;

    // Per-plane "all-ones" pattern for coord == N-1 detection.
    // mask3 = (1<<(3*logN))-1; xm_all = XM & mask3, etc.
    uint mask3  = (1u << (3u * logN)) - 1u;
    uint xm_all = XM & mask3;
    uint ym_all = YM & mask3;
    uint zm_all = ZM & mask3;

    // Branch-merged boundary test: a single OR-reduction.
    uint face_bits = (mx ^ xm_all) | (my ^ ym_all) | (mz ^ zm_all);
    // face if any of mx,my,mz is 0 OR equals its all-ones pattern.
    // The xor==0 branch catches the "==all_ones" case; the ==0 case
    // is checked separately because mx,my,mz can each independently be 0.
    bool on_face = (mx == 0u) | (my == 0u) | (mz == 0u)
                 | (mx == xm_all) | (my == ym_all) | (mz == zm_all);
    (void)face_bits;

    if (on_face) {
        u_out[tid] = u_in[tid];
        return;
    }

    // Precompute combined complement masks.
    uint YZ = my | mz;
    uint XZ = mx | mz;
    uint XY = mx | my;
    uint YM_ZM = YM | ZM;
    uint XM_ZM = XM | ZM;
    uint XM_YM = XM | YM;

    // Neighbour Morton indices via masked carry/borrow on each bit-plane.
    uint m_xp = (((m | YM_ZM) + 1u) & XM) | YZ;
    uint m_xm = ((mx - 1u) & XM) | YZ;
    uint m_yp = (((m | XM_ZM) + 2u) & YM) | XZ;
    uint m_ym = ((my - 2u) & YM) | XZ;
    uint m_zp = (((m | XM_YM) + 4u) & ZM) | XY;
    uint m_zm = ((mz - 4u) & ZM) | XY;

    // Issue all seven loads up front for maximum memory-level parallelism.
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