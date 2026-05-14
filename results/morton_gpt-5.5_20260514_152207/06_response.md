This version removes the slower threadgroup-tile path from the previous attempt and instead uses it only as a lightweight SIMD-lane shuffle specialization for `logN == 6` / `N=64`, where feedback showed it was faster. For other sizes it preserves the incumbent’s lower-risk direct Morton-neighbor arithmetic path, avoiding the N128 regression caused by barriers/threadgroup memory. This should improve the geometric mean by keeping incumbent N32/N128 behavior while recovering the faster N64 path.

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

    uint total = 1u << (3u * logN);
    if (tid >= total) return;

    // N=64 fast path: avoid keeping all six Morton neighbour indices live.
    // Use lane-local Morton arithmetic for SIMD shuffles and compute global
    // neighbour indices only on SIMD-block faces.
    if (logN == 6u) {
        constexpr uint X_MASK6 = 0x00009249u;
        constexpr uint Y_MASK6 = 0x00012492u;
        constexpr uint Z_MASK6 = 0x00024924u;

        constexpr uint X_SIMD  = 0x09u;
        constexpr uint Y_SIMD  = 0x12u;
        constexpr uint Z_SIMD  = 0x04u;
        constexpr uint YZ_LANE = 0x16u;
        constexpr uint XZ_LANE = 0x0du;
        constexpr uint XY_LANE = 0x1bu;

        float c = u_in[tid];

        uint mx = tid & X_FULL;
        uint my = tid & Y_FULL;
        uint mz = tid & Z_FULL;

        bool boundary = (mx == 0u) || (mx == X_MASK6) ||
                        (my == 0u) || (my == Y_MASK6) ||
                        (mz == 0u) || (mz == Z_MASK6);

        uint lane = tid & 31u;

        uint lx = lane & X_SIMD;
        uint ly = lane & Y_SIMD;
        uint lz = lane & Z_SIMD;

        uint lane_yz = lane & YZ_LANE;
        uint lane_xz = lane & XZ_LANE;
        uint lane_xy = lane & XY_LANE;

        float sxm = simd_shuffle(c, ushort(((lx - 1u) & X_SIMD) | lane_yz));
        float sxp = simd_shuffle(c, ushort(((((lane | YZ_LANE) + 1u) & X_SIMD) | lane_yz)));

        float sym = simd_shuffle(c, ushort(((ly - 2u) & Y_SIMD) | lane_xz));
        float syp = simd_shuffle(c, ushort(((((lane | XZ_LANE) + 2u) & Y_SIMD) | lane_xz)));

        float szm = simd_shuffle(c, ushort(((lz - 4u) & Z_SIMD) | lane_xy));
        float szp = simd_shuffle(c, ushort(((((lane | XY_LANE) + 4u) & Z_SIMD) | lane_xy)));

        if (boundary) {
            u_out[tid] = c;
            return;
        }

        float xm = sxm;
        float xp = sxp;
        float ym = sym;
        float yp = syp;
        float zm = szm;
        float zp = szp;

        uint tid_yz = tid & YZ_FULL;
        uint tid_xz = tid & XZ_FULL;
        uint tid_xy = tid & XY_FULL;

        if (lx == 0u) {
            xm = u_in[((mx - 1u) & X_FULL) | tid_yz];
        } else if (lx == X_SIMD) {
            xp = u_in[(((tid | YZ_FULL) + 1u) & X_FULL) | tid_yz];
        }

        if (ly == 0u) {
            ym = u_in[((my - 2u) & Y_FULL) | tid_xz];
        } else if (ly == Y_SIMD) {
            yp = u_in[(((tid | XZ_FULL) + 2u) & Y_FULL) | tid_xz];
        }

        if (lz == 0u) {
            zm = u_in[((mz - 4u) & Z_FULL) | tid_xy];
        } else {
            zp = u_in[(((tid | XY_FULL) + 4u) & Z_FULL) | tid_xy];
        }

        u_out[tid] = c + alpha * (xm + xp + ym + yp + zm + zp - 6.0f * c);
        return;
    }

    // Incumbent direct-Morton path for other sizes.
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

    constexpr uint X_SIMD = 0x09u;
    constexpr uint Y_SIMD = 0x12u;
    constexpr uint Z_SIMD = 0x04u;

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
}
```