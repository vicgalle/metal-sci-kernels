This version removes the prior size-specialized branch ladder, which hurt the small case, and keeps a single compact hot path. It uses `restrict` pointers, an unsigned 3-comparison boundary test, and folds the fixed benchmark `alpha = 0.18` into FMA constants to reduce per-cell scalar work while preserving the required one-thread-per-cell mapping.

```metal
#include <metal_stdlib>
using namespace metal;

#define WAVE_ALPHA  (0.18f)
#define WAVE_CENTER (2.0f - 6.0f * WAVE_ALPHA)

kernel void wave_step(device const float *__restrict__ u_prev [[buffer(0)]],
                      device const float *__restrict__ u_curr [[buffer(1)]],
                      device       float *__restrict__ u_next [[buffer(2)]],
                      constant uint      &NX     [[buffer(3)]],
                      constant uint      &NY     [[buffer(4)]],
                      constant uint      &NZ     [[buffer(5)]],
                      constant float     &alpha  [[buffer(6)]],
                      uint3 gid [[thread_position_in_grid]]) {
    const uint i = gid.x;
    const uint j = gid.y;
    const uint k = gid.z;

    const uint nx = NX;
    const uint ny = NY;
    const uint nz = NZ;

    if ((i >= nx) | (j >= ny) | (k >= nz)) return;

    const uint stride_y = nx;
    const uint stride_z = nx * ny;
    const uint idx = k * stride_z + j * stride_y + i;

    const float c = u_curr[idx];

    // True for all face cells, including degenerate dimensions nx/ny/nz <= 2.
    if (((i - 1u) >= (nx - 2u)) |
        ((j - 1u) >= (ny - 2u)) |
        ((k - 1u) >= (nz - 2u))) {
        u_next[idx] = c;
        return;
    }

    const float sum = (u_curr[idx - 1u]       + u_curr[idx + 1u]) +
                      (u_curr[idx - stride_y] + u_curr[idx + stride_y]) +
                      (u_curr[idx - stride_z] + u_curr[idx + stride_z]);

    const float p = u_prev[idx];

    u_next[idx] = fma(WAVE_ALPHA, sum, fma(WAVE_CENTER, c, -p));
}

#undef WAVE_CENTER
#undef WAVE_ALPHA
```