Replaced complex boundary bit-checks with a unified mathematical property: out-of-bounds Morton neighbors in this layout strictly produce an index `>= N^3`. This eliminates multiple bit-masks and conditional branches, replacing them with a single bitwise OR and comparison. I also simplified the Morton decrement logic to `(m_x - 1) & X_MASK`, removing the need for bitwise gap-filling during subtraction. Together with precomputing the scalar center-weight, this significantly reduces ALU instruction count. It heavily speeds up compute-bound small grids (like N=32) by lowering kernel execution time, while preserving the excellent L1/SLC cache locality and peak memory bandwidth of direct index arithmetic on larger DRAM-bound grids.

```metal
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

    // O(1) Morton arithmetic. Addition requires filling gaps with 1s to propagate carry.
    // Subtraction naturally propagates borrows through 0s, saving an instruction.
    uint m_xp = m_not_x | (((m | NOT_X) + 1u) & X_MASK);
    uint m_xm = m_not_x | ((m_x - 1u) & X_MASK);

    uint m_yp = m_not_y | (((m | NOT_Y) + 2u) & Y_MASK);
    uint m_ym = m_not_y | ((m_y - 2u) & Y_MASK);

    uint m_zp = m_not_z | (((m | NOT_Z) + 4u) & Z_MASK);
    uint m_zm = m_not_z | ((m_z - 4u) & Z_MASK);

    // Mathematical property: Any out-of-bounds coordinate causes the respective 
    // Morton index to carry/borrow into bits >= 3*logN, making it >= N^3.
    // We can evaluate the boundary for all 6 directions in a single check.
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

    float sum = xm + xp + ym + yp + zm + zp;
    float c_weight = 1.0f - 6.0f * alpha;
    
    u_out[m] = sum * alpha + c * c_weight;
}
```