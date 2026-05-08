The kernel avoids the previous shared-memory/barrier overhead and keeps the cache-friendly scalar stencil. It adds aliasing qualifiers, cheaper boundary tests, and constant-stride fast paths for the benchmark cubic sizes to reduce per-cell integer address arithmetic while preserving a generic fallback.

```metal
#include <metal_stdlib>
using namespace metal;

kernel void wave_step(device const float *__restrict__ u_prev [[buffer(0)]],
                      device const float *__restrict__ u_curr [[buffer(1)]],
                      device       float *__restrict__ u_next [[buffer(2)]],
                      constant uint      &NX     [[buffer(3)]],
                      constant uint      &NY     [[buffer(4)]],
                      constant uint      &NZ     [[buffer(5)]],
                      constant float     &alpha  [[buffer(6)]],
                      uint3 gid [[thread_position_in_grid]])
{
    const uint i = gid.x;
    const uint j = gid.y;
    const uint k = gid.z;

    const uint nx = NX;
    const uint ny = NY;
    const uint nz = NZ;

    if ((i >= nx) | (j >= ny) | (k >= nz)) {
        return;
    }

    // Specialized constant-stride paths for the measured cubic domains.
    if (nx == 64u) {
        if ((ny == 64u) & (nz == 64u)) {
            const uint idx = (k << 12) + (j << 6) + i;

            if (((i - 1u) >= 62u) | ((j - 1u) >= 62u) | ((k - 1u) >= 62u)) {
                u_next[idx] = u_curr[idx];
                return;
            }

            const float c    = u_curr[idx];
            const float prev = u_prev[idx];

            const float xm = u_curr[idx - 1u];
            const float xp = u_curr[idx + 1u];
            const float ym = u_curr[idx - 64u];
            const float yp = u_curr[idx + 64u];
            const float zm = u_curr[idx - 4096u];
            const float zp = u_curr[idx + 4096u];

            const float lap = xm + xp + ym + yp + zm + zp - 6.0f * c;
            u_next[idx] = 2.0f * c - prev + alpha * lap;
            return;
        }
    } else if (nx == 160u) {
        if ((ny == 160u) & (nz == 160u)) {
            const uint idx = k * 25600u + j * 160u + i;

            if (((i - 1u) >= 158u) | ((j - 1u) >= 158u) | ((k - 1u) >= 158u)) {
                u_next[idx] = u_curr[idx];
                return;
            }

            const float c    = u_curr[idx];
            const float prev = u_prev[idx];

            const float xm = u_curr[idx - 1u];
            const float xp = u_curr[idx + 1u];
            const float ym = u_curr[idx - 160u];
            const float yp = u_curr[idx + 160u];
            const float zm = u_curr[idx - 25600u];
            const float zp = u_curr[idx + 25600u];

            const float lap = xm + xp + ym + yp + zm + zp - 6.0f * c;
            u_next[idx] = 2.0f * c - prev + alpha * lap;
            return;
        }
    } else if (nx == 192u) {
        if ((ny == 192u) & (nz == 192u)) {
            const uint idx = k * 36864u + j * 192u + i;

            if (((i - 1u) >= 190u) | ((j - 1u) >= 190u) | ((k - 1u) >= 190u)) {
                u_next[idx] = u_curr[idx];
                return;
            }

            const float c    = u_curr[idx];
            const float prev = u_prev[idx];

            const float xm = u_curr[idx - 1u];
            const float xp = u_curr[idx + 1u];
            const float ym = u_curr[idx - 192u];
            const float yp = u_curr[idx + 192u];
            const float zm = u_curr[idx - 36864u];
            const float zp = u_curr[idx + 36864u];

            const float lap = xm + xp + ym + yp + zm + zp - 6.0f * c;
            u_next[idx] = 2.0f * c - prev + alpha * lap;
            return;
        }
    }

    // Generic fallback.
    const uint idx = (k * ny + j) * nx + i;

    if (((i - 1u) >= (nx - 2u)) |
        ((j - 1u) >= (ny - 2u)) |
        ((k - 1u) >= (nz - 2u))) {
        u_next[idx] = u_curr[idx];
        return;
    }

    const uint stride_y = nx;
    const uint stride_z = nx * ny;

    const float c    = u_curr[idx];
    const float prev = u_prev[idx];

    const float xm = u_curr[idx - 1u];
    const float xp = u_curr[idx + 1u];
    const float ym = u_curr[idx - stride_y];
    const float yp = u_curr[idx + stride_y];
    const float zm = u_curr[idx - stride_z];
    const float zp = u_curr[idx + stride_z];

    const float lap = xm + xp + ym + yp + zm + zp - 6.0f * c;
    u_next[idx] = 2.0f * c - prev + alpha * lap;
}
```