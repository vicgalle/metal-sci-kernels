(1) **Optimization**: We replace 3D threadgroup memory (which suffered from high barrier and indexing overhead) with `simd_shuffle_up`/`simd_shuffle_down`. Because threads in a SIMD group map contiguously along the fast X-axis, we can share the center cell `c` with left/right neighbors via registers. We also evaluate the Dirichlet boundary condition using a branchless bitwise OR.
(2) **Why it improves**: The previous attempt was slower than the naive incumbent because L1 cache hits are faster than barrier synchronization + shared memory reads. This new version abandons threadgroup memory to heavily leverage the L1/L2 caches for the Y and Z dimensions, while using register-level SIMD shuffles to entirely eliminate 2 out of 7 global memory loads per thread. This further reduces load/store unit pressure and pushes performance strictly beyond the cache-bound incumbent.

```metal
#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(1024)]]
kernel void wave_step(device const float *u_prev [[buffer(0)]],
                      device const float *u_curr [[buffer(1)]],
                      device       float *u_next [[buffer(2)]],
                      constant uint      &NX     [[buffer(3)]],
                      constant uint      &NY     [[buffer(4)]],
                      constant uint      &NZ     [[buffer(5)]],
                      constant float     &alpha  [[buffer(6)]],
                      uint3 gid     [[thread_position_in_grid]],
                      uint3 t_pos   [[thread_position_in_threadgroup]],
                      uint3 tg_size [[threads_per_threadgroup]],
                      uint  lane_id [[thread_index_in_simdgroup]]) {
    uint nx = NX;
    uint ny = NY;
    uint nz = NZ;

    uint i = gid.x;
    uint j = gid.y;
    uint k = gid.z;
    
    // Grid bounds check must precede the shuffle so out-of-bounds threads 
    // do not evaluate u_curr at an invalid index.
    if (i >= nx || j >= ny || k >= nz) return;

    uint idx = (k * ny + j) * nx + i;
    float c = u_curr[idx];

    // Unconditionally shuffle to share 'c' along the fast X-axis.
    // Done BEFORE the boundary early-return so source lanes remain active.
    float xm_shfl = simd_shuffle_up(c, 1);
    float xp_shfl = simd_shuffle_down(c, 1);

    // Evaluate boundary strictly with bitwise OR for branchless evaluation
    bool is_boundary = (i == 0u) | (j == 0u) | (k == 0u) | 
                       (i == nx - 1u) | (j == ny - 1u) | (k == nz - 1u);
    if (is_boundary) {
        u_next[idx] = c;
        return;
    }

    // Only active threads (non-boundary) will reach here.
    // Ensure we don't cross threadgroup rows (t_pos.x) or SIMD boundaries (lane_id).
    float xm = (t_pos.x > 0 && lane_id > 0) ? xm_shfl : u_curr[idx - 1u];
    float xp = (t_pos.x < tg_size.x - 1u && lane_id < 31) ? xp_shfl : u_curr[idx + 1u];

    uint stride_z = nx * ny;
    float ym = u_curr[idx - nx];
    float yp = u_curr[idx + nx];
    float zm = u_curr[idx - stride_z];
    float zp = u_curr[idx + stride_z];

    float lap = xm + xp + ym + yp + zm + zp - 6.0f * c;
    u_next[idx] = 2.0f * c - u_prev[idx] + alpha * lap;
}
```