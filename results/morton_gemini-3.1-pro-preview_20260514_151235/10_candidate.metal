#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(256)]]
kernel void morton_stencil(
    device const float *u_in   [[buffer(0)]],
    device       float *u_out  [[buffer(1)]],
    constant uint      &N      [[buffer(2)]],
    constant uint      &logN   [[buffer(3)]],
    constant float     &alpha  [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = 1u << (3u * logN);
    if (tid >= total) return;

    uint m = tid;
    
    // 1. Hoist center load and constants
    float c = u_in[m];
    float a = alpha;
    float c_term = c * (1.0f - 6.0f * a);

    // 2. O(1) Morton boundary-preserving arithmetic (explicitly scalarized for ILP)
    // X axis
    uint not_x = m & 0xB6DB6DB6u;
    uint m_xp  = not_x | (((m | 0xB6DB6DB6u) + 1u) & 0x49249249u);
    uint m_xm  = not_x | (((m & 0x49249249u) - 1u) & 0x49249249u);

    // Y axis
    uint not_y = m & 0x6DB6DB6Du;
    uint m_yp  = not_y | (((m | 0x6DB6DB6Du) + 2u) & 0x92492492u);
    uint m_ym  = not_y | (((m & 0x92492492u) - 2u) & 0x92492492u);

    // Z axis
    uint not_z = m & 0xDB6DB6DBu;
    uint m_zp  = not_z | (((m | 0xDB6DB6DBu) + 4u) & 0x24924924u);
    uint m_zm  = not_z | (((m & 0x24924924u) - 4u) & 0x24924924u);

    // 3. Consolidated out-of-bounds check catching all 6 boundaries in one branch
    if ((m_xp | m_xm | m_yp | m_ym | m_zp | m_zm) >= total) {
        u_out[m] = c;
        return;
    }

    // 4. Gather 6 spatial neighbors
    float xm = u_in[m_xm];
    float xp = u_in[m_xp];
    float ym = u_in[m_ym];
    float yp = u_in[m_yp];
    float zm = u_in[m_zm];
    float zp = u_in[m_zp];

    // 5. Balanced accumulation tree
    float sum_x = xm + xp;
    float sum_y = ym + yp;
    float sum_z = zm + zp;
    float sum   = (sum_x + sum_y) + sum_z;

    u_out[m] = fma(sum, a, c_term);
}