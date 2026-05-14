**Optimization:** Replace the O(logN) bit loops with O(1) magic-constant bit-spread (encode) and bit-compact (decode), and compute neighbour Morton indices directly via the masked-add trick (`((m | ~MASK) + carry) & MASK | (m & ~MASK)`), eliminating decode→encode round-trips entirely. This collapses ~6·logN ops per neighbour to a few ALU ops, freeing the kernel to be memory-bound. The Morton-ordered consecutive `tid` access pattern is preserved (boundary check still needs decoded coords, but only once and cheaply). This should help all sizes, especially the held-out 256³ where the kernel becomes memory-bound and the seed's heavy ALU was masking bandwidth headroom.

```metal
#include <metal_stdlib>
using namespace metal;

// Compact bits 0,3,6,... of v into bits 0,1,2,... (inverse of spread3).
inline uint compact3(uint v) {
    v &= 0x09249249u;
    v = (v | (v >>  2)) & 0x030C30C3u;
    v = (v | (v >>  4)) & 0x0300F00Fu;
    v = (v | (v >>  8)) & 0x030000FFu;
    v = (v | (v >> 16)) & 0x000003FFu;
    return v;
}

kernel void morton_stencil(device const float *u_in   [[buffer(0)]],
                           device       float *u_out  [[buffer(1)]],
                           constant uint      &N      [[buffer(2)]],
                           constant uint      &logN   [[buffer(3)]],
                           constant float     &alpha  [[buffer(4)]],
                           uint tid [[thread_position_in_grid]])
{
    uint total = N * N * N;
    if (tid >= total) return;

    // Full 32-bit Morton masks; we mask the result to 3*logN bits implicitly
    // via the buffer bounds (coords always stay in [0, N)).
    const uint XM = 0x09249249u; // x bits
    const uint YM = 0x12492492u; // y bits
    const uint ZM = 0x24924924u; // z bits
    const uint YZ = YM | ZM;
    const uint XZ = XM | ZM;
    const uint XY = XM | YM;

    // Decode for boundary check (cheap O(1) compact).
    uint x = compact3(tid);
    uint y = compact3(tid >> 1);
    uint z = compact3(tid >> 2);

    uint Nm1 = N - 1u;
    if (x == 0u || y == 0u || z == 0u ||
        x == Nm1 || y == Nm1 || z == Nm1) {
        u_out[tid] = u_in[tid];
        return;
    }

    // Neighbour Morton indices via masked carry/borrow propagation.
    // x+1: increment in the x-bit field, leaving y/z bits intact.
    uint m = tid;
    uint mx = m & XM, my = m & YM, mz = m & ZM;

    uint m_xp = (((m | YZ) + 1u) & XM) | my | mz;
    uint m_xm = ((mx - 1u) & XM) | my | mz;
    uint m_yp = (((m | XZ) + 2u) & YM) | mx | mz;
    uint m_ym = ((my - 2u) & YM) | mx | mz;
    uint m_zp = (((m | XY) + 4u) & ZM) | mx | my;
    uint m_zm = ((mz - 4u) & ZM) | mx | my;

    float c  = u_in[tid];
    float xm = u_in[m_xm];
    float xp = u_in[m_xp];
    float ym = u_in[m_ym];
    float yp = u_in[m_yp];
    float zm = u_in[m_zm];
    float zp = u_in[m_zp];

    u_out[tid] = c + alpha * (xm + xp + ym + yp + zm + zp - 6.0f * c);
}
```