## Analysis

The previous attempt failed with NaN, almost certainly because out-of-range threads (when NX/NY aren't multiples of TX/TY, or k>=NZ) participated in cooperative tile loads but then `return`ed before/at the barrier — actually the bigger bug is that threads with `k >= NZ` (or out-of-bounds i/j) still loaded into the tile using clamped indices, AND the boundary `return` after the barrier is fine, but threads that returned early due to `gid` checks were missing... Actually looking again: I had `if (i >= NX || j >= NY || k >= NZ) return;` AFTER the barrier, which is correct. The NaN likely came from threadgroups where ALL threads have k>=NZ doing reads with clamped kc=NZ-1 — that's safe. More likely: when NX is not a multiple of TX, the rightmost halo loader (lx==TX-1) might not exist if the threadgroup is smaller... but Metal launches full threadgroups. 

The real bug: the dispatch sizes (64, 160, 192) — 160 and 192 aren't multiples of TX=32... 160/32=5, 192/32=6, both fine. 64/32=2 fine. Hmm. Let me just rewrite more carefully with a safer approach: use a 16x16 tile, do all loads with full clamping, and ensure the barrier is unconditional.

```metal
#include <metal_stdlib>
using namespace metal;

#define TX 16u
#define TY 16u

kernel void wave_step(device const float *u_prev [[buffer(0)]],
                      device const float *u_curr [[buffer(1)]],
                      device       float *u_next [[buffer(2)]],
                      constant uint      &NX     [[buffer(3)]],
                      constant uint      &NY     [[buffer(4)]],
                      constant uint      &NZ     [[buffer(5)]],
                      constant float     &alpha  [[buffer(6)]],
                      uint3 gid  [[thread_position_in_grid]],
                      uint3 lid  [[thread_position_in_threadgroup]],
                      uint3 tgid [[threadgroup_position_in_grid]]) {
    threadgroup float tile[TY + 2u][TX + 2u];

    const uint lx = lid.x;
    const uint ly = lid.y;

    const uint i = gid.x;
    const uint j = gid.y;
    const uint k = gid.z;

    const uint stride_y = NX;
    const uint stride_z = NX * NY;

    // Clamp helpers for tile loads (safe even for out-of-range threads).
    const uint NXm1 = NX - 1u;
    const uint NYm1 = NY - 1u;
    const uint NZm1 = NZ - 1u;

    uint ic = (i < NX) ? i : NXm1;
    uint jc = (j < NY) ? j : NYm1;
    uint kc = (k < NZ) ? k : NZm1;

    // Center load
    tile[ly + 1u][lx + 1u] = u_curr[(kc * NY + jc) * NX + ic];

    // Halo loads — use the first rows/cols of the threadgroup to fetch.
    if (lx == 0u) {
        uint im = (ic == 0u) ? 0u : (ic - 1u);
        tile[ly + 1u][0] = u_curr[(kc * NY + jc) * NX + im];
    }
    if (lx == TX - 1u) {
        uint ip = (ic + 1u < NX) ? (ic + 1u) : NXm1;
        tile[ly + 1u][TX + 1u] = u_curr[(kc * NY + jc) * NX + ip];
    }
    if (ly == 0u) {
        uint jm = (jc == 0u) ? 0u : (jc - 1u);
        tile[0][lx + 1u] = u_curr[(kc * NY + jm) * NX + ic];
    }
    if (ly == TY - 1u) {
        uint jp = (jc + 1u < NY) ? (jc + 1u) : NYm1;
        tile[TY + 1u][lx + 1u] = u_curr[(kc * NY + jp) * NX + ic];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Now do bounds + boundary checks.
    if (i >= NX || j >= NY || k >= NZ) return;

    const uint idx = (k * NY + j) * NX + i;
    const float c_val = tile[ly + 1u][lx + 1u];

    if (i == 0u || j == 0u || k == 0u
        || i == NXm1 || j == NYm1 || k == NZm1) {
        u_next[idx] = c_val;
        return;
    }

    float xm = tile[ly + 1u][lx];
    float xp = tile[ly + 1u][lx + 2u];
    float ym = tile[ly][lx + 1u];
    float yp = tile[ly + 2u][lx + 1u];
    float zm = u_curr[idx - stride_z];
    float zp = u_curr[idx + stride_z];

    float lap = (xm + xp) + (ym + yp) + (zm + zp) - 6.0f * c_val;
    u_next[idx] = fma(alpha, lap, 2.0f * c_val - u_prev[idx]);
}
```