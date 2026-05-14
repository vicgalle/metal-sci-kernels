#include <metal_stdlib>
using namespace metal;

#define TG 256u

[[max_total_threads_per_threadgroup(256)]]
kernel void morton_stencil(device const float *u_in   [[buffer(0)]],
                           device       float *u_out  [[buffer(1)]],
                           constant uint      &N      [[buffer(2)]],
                           constant uint      &logN   [[buffer(3)]],
                           constant float     &alpha  [[buffer(4)]],
                           uint tid  [[thread_position_in_grid]],
                           uint ltid [[thread_position_in_threadgroup]],
                           uint gid  [[threadgroup_position_in_grid]])
{
    threadgroup float sm[TG];

    const uint XM = 0x09249249u;
    const uint YM = 0x12492492u;
    const uint ZM = 0x24924924u;

    uint total = N * N * N;
    uint base  = gid * TG;          // first Morton index handled by this TG

    // Cooperative load of TG consecutive Morton indices into threadgroup mem.
    uint load_idx = base + ltid;
    if (load_idx < total) {
        sm[ltid] = u_in[load_idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid >= total) return;

    uint m  = tid;
    uint mx = m & XM;
    uint my = m & YM;
    uint mz = m & ZM;

    // Boundary check without decoding.
    uint mask3  = (1u << (3u * logN)) - 1u;
    uint xm_all = XM & mask3;
    uint ym_all = YM & mask3;
    uint zm_all = ZM & mask3;

    if (mx == 0u || my == 0u || mz == 0u ||
        mx == xm_all || my == ym_all || mz == zm_all) {
        u_out[tid] = u_in[tid];
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

    uint top = base + TG;

    // Helper: load from shared memory if neighbour lies in this TG's
    // Morton range [base, base+TG), else from device memory.
    float c  = sm[ltid];

    float xp = (m_xp >= base && m_xp < top) ? sm[m_xp - base] : u_in[m_xp];
    float xn = (m_xm >= base && m_xm < top) ? sm[m_xm - base] : u_in[m_xm];
    float yp = (m_yp >= base && m_yp < top) ? sm[m_yp - base] : u_in[m_yp];
    float yn = (m_ym >= base && m_ym < top) ? sm[m_ym - base] : u_in[m_ym];
    float zp = (m_zp >= base && m_zp < top) ? sm[m_zp - base] : u_in[m_zp];
    float zn = (m_zm >= base && m_zm < top) ? sm[m_zm - base] : u_in[m_zm];

    float sum6 = (xp + xn) + (yp + yn) + (zp + zn);
    u_out[tid] = fma(alpha, sum6 - 6.0f * c, c);
}