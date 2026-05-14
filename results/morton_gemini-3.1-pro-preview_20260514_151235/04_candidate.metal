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

    constexpr uint X_MASK  = 0x09249249u;
    constexpr uint Y_MASK  = 0x12492492u;
    constexpr uint Z_MASK  = 0x24924924u;
    
    constexpr uint YZ_MASK = Y_MASK | Z_MASK;
    constexpr uint XZ_MASK = X_MASK | Z_MASK;
    constexpr uint XY_MASK = X_MASK | Y_MASK;

    uint m = tid;
    float c = u_in[m];

    uint valid_mask = (1u << (3u * logN)) - 1u;
    uint x_mask = X_MASK & valid_mask;
    uint y_mask = Y_MASK & valid_mask;
    uint z_mask = Z_MASK & valid_mask;

    // Fast global boundary check
    uint not_m = ~m;
    bool is_boundary = ((m & x_mask) == 0u) | ((not_m & x_mask) == 0u) |
                       ((m & y_mask) == 0u) | ((not_m & y_mask) == 0u) |
                       ((m & z_mask) == 0u) | ((not_m & z_mask) == 0u);

    // Identify local 4x4x2 block coordinates within the SIMD group
    uint lane = m & 31u;
    uint lx = (lane & 1u) | ((lane >> 2u) & 2u);
    uint ly = ((lane >> 1u) & 1u) | ((lane >> 3u) & 2u);
    uint lz = (lane >> 2u) & 1u;

    // Calculate shuffle lanes for intra-block neighbors
    uint xp_lane = (lane + ((lx == 1u) ? 7u : 1u)) & 31u;
    uint xm_lane = (lane - ((lx == 2u) ? 7u : 1u)) & 31u;
    
    uint yp_lane = (lane + ((ly == 1u) ? 14u : 2u)) & 31u;
    uint ym_lane = (lane - ((ly == 2u) ? 14u : 2u)) & 31u;
    
    uint zp_lane = (lane + 4u) & 31u;
    uint zm_lane = (lane - 4u) & 31u;

    // Unconditionally execute shuffles for uniform SIMD behavior
    float xp = simd_shuffle(c, xp_lane);
    float xm = simd_shuffle(c, xm_lane);
    float yp = simd_shuffle(c, yp_lane);
    float ym = simd_shuffle(c, ym_lane);
    float zp = simd_shuffle(c, zp_lane);
    float zm = simd_shuffle(c, zm_lane);

    // Dirichlet BC guarantees output is unmodified on cube faces
    if (!is_boundary) {
        uint m_yz = m & YZ_MASK;
        uint m_xz = m & XZ_MASK;
        uint m_xy = m & XY_MASK;

        // Masking with valid_mask guarantees safe wraparound bounds if fetched
        if (lx == 3u) xp = u_in[((((m | YZ_MASK) + 1u) & X_MASK) | m_yz) & valid_mask];
        if (lx == 0u) xm = u_in[((((m & X_MASK) - 1u) & X_MASK) | m_yz) & valid_mask];
        
        if (ly == 3u) yp = u_in[((((m | XZ_MASK) + 2u) & Y_MASK) | m_xz) & valid_mask];
        if (ly == 0u) ym = u_in[((((m & Y_MASK) - 2u) & Y_MASK) | m_xz) & valid_mask];
        
        if (lz == 1u) zp = u_in[((((m | XY_MASK) + 4u) & Z_MASK) | m_xy) & valid_mask];
        if (lz == 0u) zm = u_in[((((m & Z_MASK) - 4u) & Z_MASK) | m_xy) & valid_mask];

        float sum = xm + xp + ym + yp + zm + zp;
        u_out[m] = c + alpha * (sum - 6.0f * c);
    } else {
        u_out[m] = c;
    }
}