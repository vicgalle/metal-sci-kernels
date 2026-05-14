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
    threadgroup float tile[256];
    (void)N;

    constexpr uint X_FULL  = 0x09249249u;
    constexpr uint Y_FULL  = 0x12492492u;
    constexpr uint Z_FULL  = 0x24924924u;
    constexpr uint YZ_FULL = 0x36db6db6u;
    constexpr uint XZ_FULL = 0x2db6db6du;
    constexpr uint XY_FULL = 0x1b6db6dbu;

    constexpr uint X_SIMD  = 0x09u;
    constexpr uint Y_SIMD  = 0x12u;
    constexpr uint Z_SIMD  = 0x04u;
    constexpr uint YZ_LANE = 0x16u;
    constexpr uint XZ_LANE = 0x0du;
    constexpr uint XY_LANE = 0x1bu;

    constexpr uint X_TG = 0x49u;
    constexpr uint Y_TG = 0x92u;
    constexpr uint Z_TG = 0x24u;

    uint total = 1u << (3u * logN);
    if (tid >= total) return;

    float c = u_in[tid];

    // Small/SLC-resident sizes: avoid threadgroup barrier overhead.
    if (logN <= 6u) {
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

        uint lane = tid & 31u;

        float sxm = simd_shuffle(c, ushort((((lane & X_SIMD) - 1u) & X_SIMD) | (lane & YZ_LANE)));
        float sxp = simd_shuffle(c, ushort(((((lane | YZ_LANE) + 1u) & X_SIMD) | (lane & YZ_LANE))));

        float sym = simd_shuffle(c, ushort((((lane & Y_SIMD) - 2u) & Y_SIMD) | (lane & XZ_LANE)));
        float syp = simd_shuffle(c, ushort(((((lane | XZ_LANE) + 2u) & Y_SIMD) | (lane & XZ_LANE))));

        float szm = simd_shuffle(c, ushort((((lane & Z_SIMD) - 4u) & Z_SIMD) | (lane & XY_LANE)));
        float szp = simd_shuffle(c, ushort(((((lane | XY_LANE) + 4u) & Z_SIMD) | (lane & XY_LANE))));

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

        uint lx = tid & X_SIMD;
        if (lx == 0u) {
            uint tid_yz = tid & YZ_FULL;
            uint m = ((mx - 1u) & X_FULL) | tid_yz;
            xm = u_in[m];
        } else if (lx == X_SIMD) {
            uint tid_yz = tid & YZ_FULL;
            uint m = (((tid | YZ_FULL) + 1u) & X_FULL) | tid_yz;
            xp = u_in[m];
        }

        uint ly = tid & Y_SIMD;
        if (ly == 0u) {
            uint tid_xz = tid & XZ_FULL;
            uint m = ((my - 2u) & Y_FULL) | tid_xz;
            ym = u_in[m];
        } else if (ly == Y_SIMD) {
            uint tid_xz = tid & XZ_FULL;
            uint m = (((tid | XZ_FULL) + 2u) & Y_FULL) | tid_xz;
            yp = u_in[m];
        }

        uint lz = tid & Z_SIMD;
        if (lz == 0u) {
            uint tid_xy = tid & XY_FULL;
            uint m = ((mz - 4u) & Z_FULL) | tid_xy;
            zm = u_in[m];
        } else {
            uint tid_xy = tid & XY_FULL;
            uint m = (((tid | XY_FULL) + 4u) & Z_FULL) | tid_xy;
            zp = u_in[m];
        }

        u_out[tid] = c + alpha * (xm + xp + ym + yp + zm + zp - 6.0f * c);
        return;
    }

    // Larger grids: cache the aligned 256-element Morton block in threadgroup memory.
    tile[tid & 255u] = c;
    threadgroup_barrier(mem_flags::mem_threadgroup);

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

    uint lane = tid & 31u;

    float sxm = simd_shuffle(c, ushort((((lane & X_SIMD) - 1u) & X_SIMD) | (lane & YZ_LANE)));
    float sxp = simd_shuffle(c, ushort(((((lane | YZ_LANE) + 1u) & X_SIMD) | (lane & YZ_LANE))));

    float sym = simd_shuffle(c, ushort((((lane & Y_SIMD) - 2u) & Y_SIMD) | (lane & XZ_LANE)));
    float syp = simd_shuffle(c, ushort(((((lane | XZ_LANE) + 2u) & Y_SIMD) | (lane & XZ_LANE))));

    float szm = simd_shuffle(c, ushort((((lane & Z_SIMD) - 4u) & Z_SIMD) | (lane & XY_LANE)));
    float szp = simd_shuffle(c, ushort(((((lane | XY_LANE) + 4u) & Z_SIMD) | (lane & XY_LANE))));

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

    uint lx = tid & X_SIMD;
    if (lx == 0u) {
        uint tid_yz = tid & YZ_FULL;
        uint m = ((mx - 1u) & X_FULL) | tid_yz;
        if ((tid & X_TG) != 0u) {
            xm = tile[m & 255u];
        } else {
            xm = u_in[m];
        }
    } else if (lx == X_SIMD) {
        uint tid_yz = tid & YZ_FULL;
        uint m = (((tid | YZ_FULL) + 1u) & X_FULL) | tid_yz;
        if ((tid & X_TG) != X_TG) {
            xp = tile[m & 255u];
        } else {
            xp = u_in[m];
        }
    }

    uint ly = tid & Y_SIMD;
    if (ly == 0u) {
        uint tid_xz = tid & XZ_FULL;
        uint m = ((my - 2u) & Y_FULL) | tid_xz;
        if ((tid & Y_TG) != 0u) {
            ym = tile[m & 255u];
        } else {
            ym = u_in[m];
        }
    } else if (ly == Y_SIMD) {
        uint tid_xz = tid & XZ_FULL;
        uint m = (((tid | XZ_FULL) + 2u) & Y_FULL) | tid_xz;
        if ((tid & Y_TG) != Y_TG) {
            yp = tile[m & 255u];
        } else {
            yp = u_in[m];
        }
    }

    uint lz = tid & Z_SIMD;
    if (lz == 0u) {
        uint tid_xy = tid & XY_FULL;
        uint m = ((mz - 4u) & Z_FULL) | tid_xy;
        if ((tid & Z_TG) != 0u) {
            zm = tile[m & 255u];
        } else {
            zm = u_in[m];
        }
    } else {
        uint tid_xy = tid & XY_FULL;
        uint m = (((tid | XY_FULL) + 4u) & Z_FULL) | tid_xy;
        if ((tid & Z_TG) != Z_TG) {
            zp = tile[m & 255u];
        } else {
            zp = u_in[m];
        }
    }

    u_out[tid] = c + alpha * (xm + xp + ym + yp + zm + zp - 6.0f * c);
}