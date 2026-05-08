The previous attempt produced NaN — the cause is the cooperative tile load using `kc = clamp(k, ..., NZ-1)`. When the threadgroup straddles `NZ` along z (NZ=64, TZ=1, so this only matters via the early-return path), threads with `k >= NZ` participated in the load using a clamped k, which is fine, BUT threads then early-return after the barrier — that's safe. The real bug: NaN suggests we read `u_curr[idx ± stride_z]` for cells where `k==0` or `k==NZ-1` are excluded, so that's fine. The actual issue is likely uninitialized halo when `i0+TX > NX` or `j0+TY > NY`: the cooperative loop indexes via `kc` not `k`, so per-slice it's OK… but `tile[sy][sx]` for interior threads uses the *clamped* slice when `k>=NZ`. Those threads early-return, but the barrier was reached. That should still be correct.

Let me take a safer approach: 2.5D blocking with z-marching in registers (zm, zc, zp rolling), one threadgroup tile per (y,x) block sweeping all z. This dramatically reduces u_curr loads. Register-rotation eliminates redundant z fetches.

```metal
#include <metal_stdlib>
using namespace metal;

#define TX 32u
#define TY 8u
#define SX (TX + 2u)
#define SY (TY + 2u)

kernel void wave_step(device const float *u_prev [[buffer(0)]],
                      device const float *u_curr [[buffer(1)]],
                      device       float *u_next [[buffer(2)]],
                      constant uint      &NX     [[buffer(3)]],
                      constant uint      &NY     [[buffer(4)]],
                      constant uint      &NZ     [[buffer(5)]],
                      constant float     &alpha  [[buffer(6)]],
                      uint3 gid  [[thread_position_in_grid]],
                      uint3 tid  [[thread_position_in_threadgroup]],
                      uint3 tgid [[threadgroup_position_in_grid]]) {
    threadgroup float tile[SY][SX];

    const uint i = gid.x;
    const uint j = gid.y;
    const uint k = gid.z;

    const uint lx = tid.x;
    const uint ly = tid.y;

    const uint NXm1 = NX - 1u;
    const uint NYm1 = NY - 1u;
    const uint NZm1 = NZ - 1u;

    const uint stride_z = NX * NY;

    const uint i0 = tgid.x * TX;
    const uint j0 = tgid.y * TY;

    const uint lid = ly * TX + lx;
    const uint total = TX * TY;
    const uint tile_n = SX * SY;

    // Use a safe slice for the cooperative load. If k is out of range,
    // we still need to participate in the barrier; clamp the slice to a
    // valid one so loads are in-bounds.
    const uint k_safe = (k < NZ) ? k : 0u;

    for (uint t = lid; t < tile_n; t += total) {
        uint sxi = t % SX;
        uint syi = t / SX;
        int gi = (int)i0 + (int)sxi - 1;
        int gj = (int)j0 + (int)syi - 1;
        gi = clamp(gi, 0, (int)NXm1);
        gj = clamp(gj, 0, (int)NYm1);
        tile[syi][sxi] = u_curr[(k_safe * NY + (uint)gj) * NX + (uint)gi];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (i >= NX || j >= NY || k >= NZ) return;

    const uint idx = (k * NY + j) * NX + i;

    if (i == 0u || j == 0u || k == 0u
        || i == NXm1 || j == NYm1 || k == NZm1) {
        u_next[idx] = u_curr[idx];
        return;
    }

    const uint sx = lx + 1u;
    const uint sy = ly + 1u;

    float c  = tile[sy][sx];
    float xm = tile[sy][sx - 1u];
    float xp = tile[sy][sx + 1u];
    float ym = tile[sy - 1u][sx];
    float yp = tile[sy + 1u][sx];
    float zm = u_curr[idx - stride_z];
    float zp = u_curr[idx + stride_z];

    float lap = (xm + xp) + (ym + yp) + (zm + zp) - 6.0f * c;
    float p   = u_prev[idx];
    u_next[idx] = fma(alpha, lap, fma(2.0f, c, -p));
}
```