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

    constexpr uint X_SIMD = 0x09u;
    constexpr uint Y_SIMD = 0x12u;
    constexpr uint Z_SIMD = 0x04u;

    uint total = 1u << (3u * logN);
    if (tid >= total) return;

    // Start the center load early; the following integer work can overlap it.
    float c = u_in[tid];

    uint validMask = total - 1u;
    uint xmask = X_FULL & validMask;
    uint ymask = xmask << 1;
    uint zmask = xmask << 2;

    uint mx = tid & X_FULL;
    uint my = tid & Y_FULL;
    uint mz = tid & Z_FULL;

    bool boundary = (mx == 0u) | (mx == xmask) |
                    (my == 0u) | (my == ymask) |
                    (mz == 0u) | (mz == zmask);

    // Compute x/y Morton neighbours unconditionally.  Boundary lanes still
    // participate in shuffles so adjacent interior lanes can read their c.
    uint tid_yz = tid & YZ_FULL;
    uint tid_xz = tid & XZ_FULL;

    uint m_xp = (((tid | YZ_FULL) + 1u) & X_FULL) | tid_yz;
    uint m_xm = ((mx - 1u) & X_FULL) | tid_yz;

    uint m_yp = (((tid | XZ_FULL) + 2u) & Y_FULL) | tid_xz;
    uint m_ym = ((my - 2u) & Y_FULL) | tid_xz;

    float sxm = simd_shuffle(c, ushort(m_xm & 31u));
    float sxp = simd_shuffle(c, ushort(m_xp & 31u));
    float sym = simd_shuffle(c, ushort(m_ym & 31u));
    float syp = simd_shuffle(c, ushort(m_yp & 31u));

    // Within a 32-lane Morton SIMD group, z has extent 2, so exactly one
    // z-neighbour is in-lane and the other is outside the SIMD group.
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
    uint m_zm = ((mz - 4u) & Z_FULL) | tid_xy;
    uint m_zp = (((tid | XY_FULL) + 4u) & Z_FULL) | tid_xy;

    bool zlo = (lz == 0u);
    float zext = u_in[zlo ? m_zm : m_zp];

    float zm = zlo ? zext  : sz_in;
    float zp = zlo ? sz_in : zext;

    u_out[tid] = c + alpha * (xm + xp + ym + yp + zm + zp - 6.0f * c);
}