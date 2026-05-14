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
    
    // 1. Hoist the center load to hide its latency behind the integer arithmetic
    float c = u_in[m];

    constexpr uint X_MASK  = 0x49249249u; // bits 0, 3, 6, ..., 30
    constexpr uint Y_MASK  = 0x92492492u; // bits 1, 4, 7, ..., 31
    constexpr uint Z_MASK  = 0x24924924u; // bits 2, 5, 8, ..., 29
    
    constexpr uint NOT_X   = ~X_MASK;
    constexpr uint NOT_Y   = ~Y_MASK;
    constexpr uint NOT_Z   = ~Z_MASK;

    uint m_not_x = m & NOT_X;
    uint m_not_y = m & NOT_Y;
    uint m_not_z = m & NOT_Z;

    // O(1) Morton arithmetic. Addition fills gaps with 1s to propagate carry.
    // Subtraction propagates borrows through 0s.
    uint m_xp = m_not_x | (((m | NOT_X) + 1u) & X_MASK);
    uint m_xm = m_not_x | (((m & X_MASK) - 1u) & X_MASK);

    uint m_yp = m_not_y | (((m | NOT_Y) + 2u) & Y_MASK);
    uint m_ym = m_not_y | (((m & Y_MASK) - 2u) & Y_MASK);

    uint m_zp = m_not_z | (((m | NOT_Z) + 4u) & Z_MASK);
    uint m_zm = m_not_z | (((m & Z_MASK) - 4u) & Z_MASK);

    // Any out-of-bounds coordinate causes the respective Morton index to carry/borrow 
    // into bits >= 3*logN, making it >= N^3. A single bitwise OR catches all 6 boundaries.
    if ((m_xm | m_xp | m_ym | m_yp | m_zm | m_zp) >= total) {
        u_out[m] = c;
        return;
    }

    float xm = u_in[m_xm];
    float xp = u_in[m_xp];
    float ym = u_in[m_ym];
    float yp = u_in[m_yp];
    float zm = u_in[m_zm];
    float zp = u_in[m_zp];

    // 2. Balanced tree to shorten dependency chains and expose ILP
    float sum_x = xm + xp;
    float sum_y = ym + yp;
    float sum_z = zm + zp;
    float sum   = (sum_x + sum_y) + sum_z;

    // 3. One explicit multiply, one FMA instruction for the final calculation
    float a = alpha;
    float c_weight = 1.0f - 6.0f * a;
    
    u_out[m] = fma(sum, a, c * c_weight);
}