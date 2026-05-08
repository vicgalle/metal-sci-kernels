This version uses SIMDgroup shuffles to share center values between neighboring lanes, replacing many horizontal (and same-SIMDgroup vertical) global memory loads. Boundary lanes still participate in shuffles before copying through, so adjacent interior cells can safely consume their values. This should reduce load pressure versus the naïve 5-load stencil while preserving one output per thread.

```metal
#include <metal_stdlib>
using namespace metal;

kernel void heat_step(device const float *u_in  [[buffer(0)]],
                      device       float *u_out [[buffer(1)]],
                      constant uint      &NX    [[buffer(2)]],
                      constant uint      &NY    [[buffer(3)]],
                      constant float     &alpha [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]],
                      uint2 tid [[thread_position_in_threadgroup]],
                      uint2 tpg [[threads_per_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    uint i = gid.x;
    uint j = gid.y;

    if (i >= NX || j >= NY) return;

    uint idx = j * NX + i;
    float c = u_in[idx];

    // All valid lanes, including boundary lanes, participate so neighboring
    // interior lanes can read boundary values through SIMD shuffles.
    float sh_left  = simd_shuffle_up(c, ushort(1));
    float sh_right = simd_shuffle_down(c, ushort(1));

    float sh_up = 0.0f;
    float sh_down = 0.0f;
    bool can_shuffle_vertical = (tpg.x < 32u);
    if (can_shuffle_vertical) {
        ushort dx = ushort(tpg.x);
        sh_up   = simd_shuffle_up(c, dx);
        sh_down = simd_shuffle_down(c, dx);
    }

    bool interior = ((i - 1u) < (NX - 2u)) && ((j - 1u) < (NY - 2u));

    if (!interior) {
        u_out[idx] = c;
        return;
    }

    float l;
    if (tid.x != 0u && lane != 0u) {
        l = sh_left;
    } else {
        l = u_in[idx - 1u];
    }

    float r;
    if ((tid.x + 1u < tpg.x) && lane != 31u) {
        r = sh_right;
    } else {
        r = u_in[idx + 1u];
    }

    float upv;
    if (can_shuffle_vertical && tid.y != 0u && lane >= tpg.x) {
        upv = sh_up;
    } else {
        upv = u_in[idx - NX];
    }

    float downv;
    if (can_shuffle_vertical && (tid.y + 1u < tpg.y) && (lane + tpg.x < 32u)) {
        downv = sh_down;
    } else {
        downv = u_in[idx + NX];
    }

    float lap = l + r + upv + downv - 4.0f * c;
    u_out[idx] = c + alpha * lap;
}
```