**Optimization:** Threadgroup-cooperative tiling using shared memory. A 1-D dispatch with `tid = Morton index` and threadgroup width 256 = 2^8 means each threadgroup covers Morton indices `[tg*256, tg*256+256)`, which (since logN≥5) corresponds to the low 8 Morton bits — a contiguous 4×4×4 sub-cube within the cube. I load 256 cells into threadgroup memory in one coalesced pass, then for each cell determine whether each of its 6 neighbours falls inside the same 4×4×4 block (cheap: the neighbour's Morton index XOR'd against the block base has only low-8 bits set). Interior neighbours read from threadgroup memory (fast, no L1 traffic); boundary neighbours read from device. This converts ~7 device loads/cell into roughly 1 device load + threadgroup loads for the inner 2×2×2 (8 cells), and partial sharing for the outer shell. Especially helpful for the small N where launch overhead dominates and DRAM hit rate is high.

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

    // Threadgroup tile: 256 consecutive Morton indices = 4x4x4 sub-cube
    // (low 8 Morton bits vary within the tile; high bits are fixed = tgid<<8).
    threadgroup float tile[256];

    // Cooperative load. tid may be >= total at the tail; guard the load.
    float my_val = 0.0f;
    if (tid < total) {
        my_val = u_in[tid];
    }
    tile[ltid] = my_val;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid >= total) return;

    uint m  = tid;
    uint mx = m & XM;
    uint my = m & YM;
    uint mz = m & ZM;

    // Boundary check: coord==0 iff its bit-plane is 0; coord==N-1 iff its
    // bit-plane equals the masked all-ones pattern.
    uint mask3  = (1u << (3u * logN)) - 1u;
    uint xm_all = XM & mask3;
    uint ym_all = YM & mask3;
    uint zm_all = ZM & mask3;

    if (mx == 0u || my == 0u || mz == 0u ||
        mx == xm_all || my == ym_all || mz == zm_all) {
        u_out[tid] = my_val;
        return;
    }

    uint YZ = my | mz;
    uint XZ = mx | mz;
    uint XY = mx | my;

    // Neighbour Morton indices.
    uint m_xp = (((m | (YM | ZM)) + 1u) & XM) | YZ;
    uint m_xm = ((mx - 1u) & XM) | YZ;
    uint m_yp = (((m | (XM | ZM)) + 2u) & YM) | XZ;
    uint m_ym = ((my - 2u) & YM) | XZ;
    uint m_zp = (((m | (XM | YM)) + 4u) & ZM) | XY;
    uint m_zm = ((mz - 4u) & ZM) | XY;

    // A neighbour is inside the tile iff its high bits (>>8) match tgid,
    // equivalently (neighbour ^ (tgid<<8)) < 256, equivalently the top
    // 24 bits equal those of tid. Since the tile is exactly 256 Morton
    // indices, the neighbour is in-tile iff (neighbour & ~0xFFu) == (tid & ~0xFFu).
    uint base = tid & ~0xFFu;

    float xp = ((m_xp & ~0xFFu) == base) ? tile[m_xp & 0xFFu] : u_in[m_xp];
    float xn = ((m_xm & ~0xFFu) == base) ? tile[m_xm & 0xFFu] : u_in[m_xm];
    float yp = ((m_yp & ~0xFFu) == base) ? tile[m_yp & 0xFFu] : u_in[m_yp];
    float yn = ((m_ym & ~0xFFu) == base) ? tile[m_ym & 0xFFu] : u_in[m_ym];
    float zp = ((m_zp & ~0xFFu) == base) ? tile[m_zp & 0xFFu] : u_in[m_zp];
    float zn = ((m_zm & ~0xFFu) == base) ? tile[m_zm & 0xFFu] : u_in[m_zm];

    float c    = my_val;
    float sum6 = (xp + xn) + (yp + yn) + (zp + zn);
    u_out[tid] = fma(alpha, sum6 - 6.0f * c, c);
}
```