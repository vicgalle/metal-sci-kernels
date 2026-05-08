The previous attempt produced NaN — almost certainly because threads with `gid.z >= NZ` (overhang from rounded-up dispatch) returned before participating in the cooperative tile load, leaving uninitialized threadgroup memory used by other valid threads. Also threads with `gid.y >= NY` had the same issue. The fix: do the cooperative load using ALL threads in the threadgroup (don't early-return), and only guard the *write* with the bounds/boundary check. The tile is loaded from u_curr at slice `kc = clamp(k, 0, NZ-1)` so out-of-range z-threads still load valid data. The XY tile in threadgroup memory amortizes 4 of 6 neighbor loads, and Z neighbors come straight from device memory. This should be correct and faster than the incumbent on the bandwidth-bound large cases.

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

    // Clamp k for the tile load so out-of-range z-threads still cooperate
    // with valid in-bounds reads (instead of returning early and leaving
    // threadgroup memory uninitialized).
    const uint kc = (k < NZ) ? k : NZm1;

    const uint lid = ly * TX + lx;
    const uint total = TX * TY;
    const uint tile_n = SX * SY;

    // Cooperative halo load of (TY+2) x (TX+2) tile from u_curr at slice kc.
    // ALL threads participate, regardless of whether their gid is in bounds.
    for (uint t = lid; t < tile_n; t += total) {
        uint sxi = t % SX;
        uint syi = t / SX;
        int gi = (int)i0 + (int)sxi - 1;
        int gj = (int)j0 + (int)syi - 1;
        gi = clamp(gi, 0, (int)NXm1);
        gj = clamp(gj, 0, (int)NYm1);
        tile[syi][sxi] = u_curr[(kc * NY + (uint)gj) * NX + (uint)gi];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Now safe to early-out for out-of-range threads.
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