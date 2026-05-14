I specialize the kernel by grid size: for N=32 I keep the incumbent path that was fastest on the small, cache-resident case, while for larger grids I use the lower-instruction z-neighbor path from the previous attempt. This avoids the small-size regression while retaining the better N64/N128 behavior, with a uniform `logN` branch so there is no per-lane divergence. 

```metal
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

    constexpr uint X_FULL  = 0x09249249u;
    constexpr uint Y_FULL  = 0x12492492u;
    constexpr uint Z_FULL  = 0x24924924u;
    constexpr uint YZ_FULL = 0x36db6db6u;
    constexpr uint XZ_FULL = 0x2db6db6du;
    constexpr uint XY_FULL = 0x1b6db6dbu;

    constexpr uint X_SIMD = 0x09u;
    constexpr uint Y_SIMD = 0x12u;
    constexpr uint Z_SIMD = 0x04u;

    uint total = 1u << (3u * logN);
    if (tid >= total) return;

    // The incumbent path is materially faster for the small N=32 case.
    if (logN <= 5u) {
        uint validMask = total - 1u;
        uint xmask = X_FULL & validMask;
        uint ymask = xmask << 1;
        uint zmask = xmask << 2;

        uint mx = tid & X_FULL;
        uint my = tid & Y_FULL;
        uint mz = tid & Z_FULL;

        bool boundary = (mx == 0u) || (mx == xmask) ||
                        (my == 0u) || (my == ymask) ||
                        (mz == 0u) || (mz == zmask);

        float c = u_in[tid];

        uint m_xm = tid, m_xp = tid;
        uint m_ym = tid, m_yp = tid;
        uint m_zm = tid, m_zp = tid;

        if (!boundary) {
            uint tid_yz = tid & YZ_FULL;
            uint tid_xz = tid & XZ_FULL;
            uint tid_xy = tid & XY_FULL;

            m_xp = (((tid | YZ_FULL) + 1u) & X_FULL) | tid_yz;
            m_xm = ((mx - 1u) & X_FULL) | tid_yz;

            m_yp = (((tid | XZ_FULL) + 2u) & Y_FULL) | tid_xz;
            m_ym = ((my - 2u) & Y_FULL) | tid_xz;

            m_zp = (((tid | XY_FULL) + 4u) & Z_FULL) | tid_xy;
            m_zm = ((mz - 4u) & Z_FULL) | tid_xy;
        }

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

        uint lx = tid & X_SIMD;
        uint ly = tid & Y_SIMD;
        uint lz = tid & Z_SIMD;

        float xm = sxm;
        if (lx == 0u) xm = u_in[m_xm];

        float xp = sxp;
        if (lx == X_SIMD) xp = u_in[m_xp];

        float ym = sym;
        if (ly == 0u) ym = u_in[m_ym];

        float yp = syp;
        if (ly == Y_SIMD) yp = u_in[m_yp];

        float zm = szm;
        if (lz == 0u) zm = u_in[m_zm];

        float zp = szp;
        if (lz == Z_SIMD) zp = u_in[m_zp];

        u_out[tid] = c + alpha * (xm + xp + ym + yp + zm + zp - 6.0f * c);
        return;
    }

    // Larger grids benefit from avoiding the second z shuffle and computing
    // only the single out-of-SIMD z neighbor needed by each lane.
    float c = u_in[tid];

    uint validMask = total - 1u;
    uint xmask = X_FULL & validMask;
    uint ymask = xmask << 1;
    uint zmask = xmask << 2;

    uint mx = tid & X_FULL;
    uint my = tid & Y_FULL;
    uint mz = tid & Z_FULL;

    bool boundary = (mx == 0u) || (mx == xmask) ||
                    (my == 0u) || (my == ymask) ||
                    (mz == 0u) || (mz == zmask);

    uint m_xm = tid, m_xp = tid;
    uint m_ym = tid, m_yp = tid;

    if (!boundary) {
        uint tid_yz = tid & YZ_FULL;
        uint tid_xz = tid & XZ_FULL;

        m_xp = (((tid | YZ_FULL) + 1u) & X_FULL) | tid_yz;
        m_xm = ((mx - 1u) & X_FULL) | tid_yz;

        m_yp = (((tid | XZ_FULL) + 2u) & Y_FULL) | tid_xz;
        m_ym = ((my - 2u) & Y_FULL) | tid_xz;
    }

    float sxm = simd_shuffle(c, ushort(m_xm & 31u));
    float sxp = simd_shuffle(c, ushort(m_xp & 31u));
    float sym = simd_shuffle(c, ushort(m_ym & 31u));
    float syp = simd_shuffle(c, ushort(m_yp & 31u));

    float sz_in = simd_shuffle(c, ushort((tid ^ Z_SIMD) & 31u));

    if (boundary) {
        u_out[tid] = c;
        return;
    }

    uint lx = tid & X_SIMD;
    uint ly = tid & Y_SIMD;
    uint lz = tid & Z_SIMD;

    float xm = sxm;
    if (lx == 0u) xm = u_in[m_xm];

    float xp = sxp;
    if (lx == X_SIMD) xp = u_in[m_xp];

    float ym = sym;
    if (ly == 0u) ym = u_in[m_ym];

    float yp = syp;
    if (ly == Y_SIMD) yp = u_in[m_yp];

    uint tid_xy = tid & XY_FULL;
    uint m_zext;
    if (lz == 0u) {
        m_zext = ((mz - 4u) & Z_FULL) | tid_xy;
    } else {
        m_zext = (((tid | XY_FULL) + 4u) & Z_FULL) | tid_xy;
    }

    float zext = u_in[m_zext];
    float zm = (lz == 0u) ? zext  : sz_in;
    float zp = (lz == 0u) ? sz_in : zext;

    u_out[tid] = c + alpha * (xm + xp + ym + yp + zm + zp - 6.0f * c);
}
```