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
    uint total = N * N * N;
    if (tid >= total) return;

    uint m = tid;
    float c = u_in[m];

    // Extended 32-bit Morton masks covering all bits
    constexpr uint X_MASK  = 0x49249249u; // bits 0, 3, 6, ..., 30
    constexpr uint Y_MASK  = 0x92492492u; // bits 1, 4, 7, ..., 31
    constexpr uint Z_MASK  = 0x24924924u; // bits 2, 5, 8, ..., 29
    
    constexpr uint NOT_X   = ~X_MASK;
    constexpr uint NOT_Y   = ~Y_MASK;
    constexpr uint NOT_Z   = ~Z_MASK;

    uint m_x = m & X_MASK;
    uint m_y = m & Y_MASK;
    uint m_z = m & Z_MASK;

    uint m_not_x = m & NOT_X;
    uint m_not_y = m & NOT_Y;
    uint m_not_z = m & NOT_Z;

    uint m_xp = m_not_x | (((m | NOT_X) + 1u) & X_MASK);
    uint m_xm = m_not_x | ((m_x - 1u) & X_MASK);

    uint m_yp = m_not_y | (((m | NOT_Y) + 2u) & Y_MASK);
    uint m_ym = m_not_y | ((m_y - 2u) & Y_MASK);

    uint m_zp = m_not_z | (((m | NOT_Z) + 4u) & Z_MASK);
    uint m_zm = m_not_z | ((m_z - 4u) & Z_MASK);

    // Share values within the 4x4x2 SIMD group using single-cycle shuffles.
    // This MUST be done unconditionally before any early-exits so all lanes are active.
    float xm = simd_shuffle(c, m_xm & 31u);
    float xp = simd_shuffle(c, m_xp & 31u);
    float ym = simd_shuffle(c, m_ym & 31u);
    float yp = simd_shuffle(c, m_yp & 31u);
    float zm = simd_shuffle(c, m_zm & 31u);
    float zp = simd_shuffle(c, m_zp & 31u);

    // Mathematical property: Any out-of-bounds coordinate causes the respective 
    // Morton index to carry/borrow into bits >= 3*logN, making it >= N^3.
    // We evaluate the global domain boundary for all 6 directions branchlessly.
    if ((m_xm | m_xp | m_ym | m_yp | m_zm | m_zp) >= total) {
        u_out[m] = c;
        return;
    }

    // For coordinates that cross the SIMD boundaries, fall back to memory.
    // The previous early-exit guarantees m_** < total here, so these loads are completely safe.
    if ((m & 9u) == 0u)   xm = u_in[m_xm]; // local x == 0
    if ((m & 9u) == 9u)   xp = u_in[m_xp]; // local x == 3
    if ((m & 18u) == 0u)  ym = u_in[m_ym]; // local y == 0
    if ((m & 18u) == 18u) yp = u_in[m_yp]; // local y == 3
    if ((m & 4u) == 0u)   zm = u_in[m_zm]; // local z == 0
    if ((m & 4u) == 4u)   zp = u_in[m_zp]; // local z == 1

    float sum = xm + xp + ym + yp + zm + zp;
    float c_weight = 1.0f - 6.0f * alpha;
    
    u_out[m] = sum * alpha + c * c_weight;
}