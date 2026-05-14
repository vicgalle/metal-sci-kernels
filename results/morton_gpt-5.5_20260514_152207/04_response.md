I keep the incumbent fast Morton-mask arithmetic and SIMD shuffle reuse for smaller grids, then switch to a 256-thread Morton-block threadgroup tile for larger grids. This preserves consecutive Morton traversal while serving cross-SIMD neighbors from threadgroup memory instead of reissuing cached/global loads, which should help most as working sets grow toward DRAM-bound sizes.

```metal
#include <metal_stdlib>
using namespace metal;

kernel void morton_stencil(
    device const float *u_in   [[buffer(0)]],
    device       float *u_out  [[buffer(1)]],
    constant uint      &N      [[buffer(2)]],
    constant uint      &logN   [[buffer(3)]],
    constant float     &alpha  [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) [[max_total_threads_per_threadgroup(256)]]
{
    threadgroup float tile[256];
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

    // For 32/64, avoid the barrier/tile overhead and use the proven SIMD-local path.
    if (logN <= 6u) {
        if (tid >= total) return;

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

    // Large grids: one 256-thread TG covers one aligned Morton block
    // with low 8 Morton bits varying: 8 x 8 x 4 cells.
    uint ltid = tid & 255u;
    bool valid = (tid < total);

    float c = 0.0f;
    if (valid) {
        c = u_in[tid];
    }

    tile[ltid] = c;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!valid) return;

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

    constexpr uint X_TG = 0x49u;
    constexpr uint Y_TG = 0x92u;
    constexpr uint Z_TG = 0x24u;

    uint lx = tid & X_SIMD;
    uint ly = tid & Y_SIMD;
    uint lz = tid & Z_SIMD;

    uint lx8 = tid & X_TG;
    uint ly8 = tid & Y_TG;
    uint lz8 = tid & Z_TG;

    float xm = sxm;
    if (lx == 0u) {
        xm = (lx8 != 0u) ? tile[m_xm & 255u] : u_in[m_xm];
    }

    float xp = sxp;
    if (lx == X_SIMD) {
        xp = (lx8 != X_TG) ? tile[m_xp & 255u] : u_in[m_xp];
    }

    float ym = sym;
    if (ly == 0u) {
        ym = (ly8 != 0u) ? tile[m_ym & 255u] : u_in[m_ym];
    }

    float yp = syp;
    if (ly == Y_SIMD) {
        yp = (ly8 != Y_TG) ? tile[m_yp & 255u] : u_in[m_yp];
    }

    float zm = szm;
    if (lz == 0u) {
        zm = (lz8 != 0u) ? tile[m_zm & 255u] : u_in[m_zm];
    }

    float zp = szp;
    if (lz == Z_SIMD) {
        zp = (lz8 != Z_TG) ? tile[m_zp & 255u] : u_in[m_zp];
    }

    u_out[tid] = c + alpha * (xm + xp + ym + yp + zm + zp - 6.0f * c);
}
```