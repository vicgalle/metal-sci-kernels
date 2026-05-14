#include <metal_stdlib>
using namespace metal;

constexpr uint X_MASK = 0x09249249u;
constexpr uint Y_MASK = 0x12492492u;
constexpr uint Z_MASK = 0x24924924u;

constexpr uint YZ_MASK = Y_MASK | Z_MASK;
constexpr uint XZ_MASK = X_MASK | Z_MASK;
constexpr uint XY_MASK = X_MASK | Y_MASK;

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

    // Compute exactly which bits are valid for the current grid size 
    uint valid_mask = (1u << (3u * logN)) - 1u;
    
    uint x_mask = X_MASK & valid_mask;
    uint y_mask = Y_MASK & valid_mask;
    uint z_mask = Z_MASK & valid_mask;

    uint m_x = m & x_mask;
    uint m_y = m & y_mask;
    uint m_z = m & z_mask;

    // Dirichlet BC: boundaries mean a dimension is either at 0 (all respective bits 0) 
    // or at N - 1 (all respective valid bits 1).
    if (m_x == 0u || m_x == x_mask ||
        m_y == 0u || m_y == y_mask ||
        m_z == 0u || m_z == z_mask) {
        u_out[m] = c;
        return;
    }

    uint m_yz = m & YZ_MASK;
    uint m_xz = m & XZ_MASK;
    uint m_xy = m & XY_MASK;

    // Direct Morton neighbour index arithmetic
    uint m_xp = (((m | YZ_MASK) + 1u) & X_MASK) | m_yz;
    uint m_xm = (((m & X_MASK) - 1u) & X_MASK) | m_yz;

    uint m_yp = (((m | XZ_MASK) + 2u) & Y_MASK) | m_xz;
    uint m_ym = (((m & Y_MASK) - 2u) & Y_MASK) | m_xz;

    uint m_zp = (((m | XY_MASK) + 4u) & Z_MASK) | m_xy;
    uint m_zm = (((m & Z_MASK) - 4u) & Z_MASK) | m_xy;

    float xm = u_in[m_xm];
    float xp = u_in[m_xp];
    float ym = u_in[m_ym];
    float yp = u_in[m_yp];
    float zm = u_in[m_zm];
    float zp = u_in[m_zp];

    u_out[m] = c + alpha * (xm + xp + ym + yp + zm + zp - 6.0f * c);
}