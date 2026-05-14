**Optimization:** Threadgroup-memory tiling for a 256-thread block. Since consecutive Morton indices cluster spatially (a 256-thread block = 8×4×8 sub-cube for logN≥3), I cooperatively load the 256 center values into shared memory. For each cell, I check if all 6 neighbours lie within the same Morton-aligned 256-block — i.e. the neighbour's high bits (above bit 7, since 256 = 2^8 covers Morton bits 0..7 = coords' low ~3 bits each) match the block's. If so, fetch from threadgroup memory; else fall back to device memory. This cuts redundant device loads for ~the interior of each block and exploits SLC/L1 reuse, especially at N=128 and the held-out N=256 where DRAM bandwidth dominates.

Actually, simpler and safer: just load the block tile into shared memory and use it opportunistically. The boundary check uses the same neighbour-arithmetic, and the "in-block" test is `(neighbour ^ tid) < 256`. Threadgroup memory is 1 KiB per block — trivial.

```metal
#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(256)]]
kernel void morton_stencil(device const float *u_in   [[buffer(0)]],
                           device       float *u_out  [[buffer(1)]],
                           constant uint      &N      [[buffer(2)]],
                           constant uint      &logN   [[buffer(3)]],
                           constant float     &alpha  [[buffer(4)]],
                           uint tid  [[thread_position_in_grid]],
                           uint ltid [[thread_position_in_threadgroup]],
                           uint tgid [[threadgroup_position_in_grid]])
{
    const uint XM = 0x09249249u;
    const uint YM = 0x12492492u;
    const uint ZM = 0x24924924u;

    uint total = N * N * N;

    // Cooperative tile load: 256 contiguous Morton indices per threadgroup.
    threadgroup float tile[256];
    float self = (tid < total) ? u_in[tid] : 0.0f;
    tile[ltid] = self;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid >= total) return;

    uint m  = tid;
    uint mx = m & XM;
    uint my = m & YM;
    uint mz = m & ZM;

    uint mask3 = (1u << (3u * logN)) - 1u;
    uint xm_all = XM & mask3;
    uint ym_all = YM & mask3;
    uint zm_all = ZM & mask3;

    if (mx == 0u || my == 0u || mz == 0u ||
        mx == xm_all || my == ym_all || mz == zm_all) {
        u_out[tid] = self;
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

    // Block base: tid & ~255. If neighbour shares the same high bits, it's in-tile.
    uint block_base = tid & ~255u;

    float xp = ((m_xp ^ tid) < 256u && (m_xp & ~255u) == block_base)
                ? tile[m_xp - block_base] : u_in[m_xp];
    float xn = ((m_xm ^ tid) < 256u && (m_xm & ~255u) == block_base)
                ? tile[m_xm - block_base] : u_in[m_xm];
    float yp = ((m_yp ^ tid) < 256u && (m_yp & ~255u) == block_base)
                ? tile[m_yp - block_base] : u_in[m_yp];
    float yn = ((m_ym ^ tid) < 256u && (m_ym & ~255u) == block_base)
                ? tile[m_ym - block_base] : u_in[m_ym];
    float zp = ((m_zp ^ tid) < 256u && (m_zp & ~255u) == block_base)
                ? tile[m_zp - block_base] : u_in[m_zp];
    float zn = ((m_zm ^ tid) < 256u && (m_zm & ~255u) == block_base)
                ? tile[m_zm - block_base] : u_in[m_zm];

    float sum6 = (xp + xn) + (yp + yn) + (zp + zn);
    u_out[tid] = fma(alpha, sum6 - 6.0f * self, self);
}
```