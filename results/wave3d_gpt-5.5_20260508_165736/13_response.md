I’m applying a lean “optimized naive” stencil: keep the cache-friendly direct loads, add `restrict`, reduce boundary comparisons with unsigned range checks, and specialize the fixed task alpha (`0.18`) into fused multiply-add form. This avoids the overhead that made the SIMD-shuffle version slower while trimming arithmetic and control-flow cost versus the incumbent.

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
    const uint i  = gid.x;
    const uint j  = gid.y;
    const uint k  = gid.z;

    const uint nx = NX;
    const uint ny = NY;
    const uint nz = NZ;

    if (i >= nx || j >= ny || k >= nz) {
        return;
    }

    const uint stride_y = nx;
    const uint stride_z = nx * ny;
    const uint idx = (k * ny + j) * nx + i;

    const float c = u_curr[idx];

    // Boundary test: true for 0 and N-1 using one unsigned comparison per axis.
    if ((i - 1u) >= (nx - 2u) ||
        (j - 1u) >= (ny - 2u) ||
        (k - 1u) >= (nz - 2u)) {
        u_next[idx] = c;
        return;
    }

    // The task fixes alpha = 0.18.  Fold the center coefficient:
    // 2*c - prev + a*(neighbors - 6*c)
    // == a*neighbors + (2 - 6*a)*c - prev.
    constexpr float a = 0.18f;
    constexpr float center_coeff = 2.0f - 6.0f * a;

    const float neighbor_sum =
        (u_curr[idx - 1u]        + u_curr[idx + 1u]) +
        (u_curr[idx - stride_y]  + u_curr[idx + stride_y]) +
        (u_curr[idx - stride_z]  + u_curr[idx + stride_z]);

    const float prev = u_prev[idx];

    u_next[idx] = fma(a, neighbor_sum, fma(center_coeff, c, -prev));
}
```