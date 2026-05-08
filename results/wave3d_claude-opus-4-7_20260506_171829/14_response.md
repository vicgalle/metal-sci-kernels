The previous attempt likely produced NaN because the cooperative tile load used `k_safe = 0` for out-of-range threads, but more critically, threads where `k >= NZ` would still execute the load loop — that's fine since clamped, but the real issue may be that the threadgroup size dispatched by the host doesn't match my assumed (TX, TY, 1) = (32, 8, 1). The host dispatches `threadsPerGrid = (NX, NY, NZ)` and Metal picks the threadgroup size automatically unless we constrain it. Without a `[[max_total_threads_per_threadgroup]]` and matching threadgroup size, my hard-coded TX/TY tile dimensions won't align with the actual threadgroup, causing wrong tile coordinates and reading garbage/uninitialized memory → NaN propagation across timesteps.

**Fix**: Avoid threadgroup memory entirely and rely on a clean register-based 7-point stencil. Use `fma` for fused arithmetic and ensure boundary handling is correct. This matches the incumbent's structure but with tighter codegen — and avoids the threadgroup-size mismatch trap. The incumbent at larger sizes is already at ~78% of bandwidth; small tweaks to ALU pattern won't hurt.

```metal
#include <metal_stdlib>
using namespace metal;

kernel void wave_step(device const float *u_prev [[buffer(0)]],
                      device const float *u_curr [[buffer(1)]],
                      device       float *u_next [[buffer(2)]],
                      constant uint      &NX     [[buffer(3)]],
                      constant uint      &NY     [[buffer(4)]],
                      constant uint      &NZ     [[buffer(5)]],
                      constant float     &alpha  [[buffer(6)]],
                      uint3 gid [[thread_position_in_grid]]) {
    const uint i = gid.x;
    const uint j = gid.y;
    const uint k = gid.z;
    if (i >= NX || j >= NY || k >= NZ) return;

    const uint NXm1 = NX - 1u;
    const uint NYm1 = NY - 1u;
    const uint NZm1 = NZ - 1u;

    const uint stride_y = NX;
    const uint stride_z = NX * NY;
    const uint idx = (k * NY + j) * NX + i;

    if (i == 0u || j == 0u || k == 0u
        || i == NXm1 || j == NYm1 || k == NZm1) {
        u_next[idx] = u_curr[idx];
        return;
    }

    float c  = u_curr[idx];
    float xm = u_curr[idx - 1u];
    float xp = u_curr[idx + 1u];
    float ym = u_curr[idx - stride_y];
    float yp = u_curr[idx + stride_y];
    float zm = u_curr[idx - stride_z];
    float zp = u_curr[idx + stride_z];
    float p  = u_prev[idx];

    float sxy = (xm + xp) + (ym + yp);
    float sz  = zm + zp;
    float lap = (sxy + sz) - 6.0f * c;

    u_next[idx] = fma(alpha, lap, fma(2.0f, c, -p));
}
```