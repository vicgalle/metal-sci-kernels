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