I’m adding a 256-cell threadgroup tile for N≥64 so neighboring cells within the Morton block are read from threadgroup memory instead of reloaded from device memory. For N=32, the kernel keeps a lighter SIMD-shuffle path to avoid barrier overhead. This should improve larger/DRAM-bound cases by extending reuse beyond one SIMDgroup while preserving the previous low-overhead behavior for small grids.

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
    (void)N;

    constexpr uint X_FULL  = 0x09249249u;
    constexpr uint Y_FULL  = 0x12492492u;
    constexpr uint Z_FULL  = 0x24924924u;
    constexpr uint YZ_FULL = 0x36db6db6u;
    constexpr uint XZ_FULL = 0x2db6db6du;
    constexpr uint XY_FULL = 0x1b6db6dbu;

    uint total = 1u << (3u * logN);

    threadgroup float tile[256];

    // For the smallest benchmark size, avoid the threadgroup barrier and use
    // only SIMD-lane reuse. Boundary lanes stay active through the shuffles.
    if (logN <= 5u) {
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
            m_xm = (((tid & X_FULL) - 1u) & X_FULL) | tid_yz;

            m_yp = (((tid | XZ_FULL) + 2u) & Y_FULL) | tid_xz;
            m_ym = (((tid & Y_FULL) - 2u) & Y_FULL) | tid_xz;

            m_zp = (((tid | XY_FULL) + 4u) & Z_FULL) | tid_xy;
            m_zm = (((tid & Z_FULL) - 4u) & Z_FULL) | tid_xy;
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

        float lap = xm + xp + ym + yp + zm + zp - 6.0f * c;
        u_out[tid] = c + alpha * lap;
        return;
    }

    // Larger sizes: one 256-thread group is one aligned Morton block.
    // Cache centers in threadgroup memory and use device memory only when a
    // stencil edge crosses the 256-element Morton block boundary.
    uint lid = tid & 255u;
    bool valid = (tid < total);

    float c = 0.0f;
    if (valid) {
        c = u_in[tid];
    }
    tile[lid] = c;

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

    if (boundary) {
        u_out[tid] = c;
        return;
    }

    uint tid_yz = tid & YZ_FULL;
    uint tid_xz = tid & XZ_FULL;
    uint tid_xy = tid & XY_FULL;

    uint m_xp = (((tid | YZ_FULL) + 1u) & X_FULL) | tid_yz;
    uint m_xm = (((tid & X_FULL) - 1u) & X_FULL) | tid_yz;

    uint m_yp = (((tid | XZ_FULL) + 2u) & Y_FULL) | tid_xz;
    uint m_ym = (((tid & Y_FULL) - 2u) & Y_FULL) | tid_xz;

    uint m_zp = (((tid | XY_FULL) + 4u) & Z_FULL) | tid_xy;
    uint m_zm = (((tid & Z_FULL) - 4u) & Z_FULL) | tid_xy;

    constexpr uint X_TG = 0x49u;
    constexpr uint Y_TG = 0x92u;
    constexpr uint Z_TG = 0x24u;

    uint lx = tid & X_TG;
    uint ly = tid & Y_TG;
    uint lz = tid & Z_TG;

    float xm = tile[m_xm & 255u];
    if (lx == 0u) xm = u_in[m_xm];

    float xp = tile[m_xp & 255u];
    if (lx == X_TG) xp = u_in[m_xp];

    float ym = tile[m_ym & 255u];
    if (ly == 0u) ym = u_in[m_ym];

    float yp = tile[m_yp & 255u];
    if (ly == Y_TG) yp = u_in[m_yp];

    float zm = tile[m_zm & 255u];
    if (lz == 0u) zm = u_in[m_zm];

    float zp = tile[m_zp & 255u];
    if (lz == Z_TG) zp = u_in[m_zp];

    float lap = xm + xp + ym + yp + zm + zp - 6.0f * c;
    u_out[tid] = c + alpha * lap;
}
```