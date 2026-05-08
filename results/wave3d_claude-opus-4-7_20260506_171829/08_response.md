The previous attempt failed because halo loads used `lx == TX-1` and `ly == TY-1`, but threads at those positions may have returned via the `if (i >= NX || j >= NY || k >= NZ) return;` guard before reaching the barrier — actually the guard was after the barrier so that's fine, but the real issue is different: when `lx==0` and `ic==0`, I wrote `tile[sy][0] = u_curr[base_kj + 0]`, which is u_curr[i=0]. But if the threadgroup straddles into out-of-domain region, threads with `i >= NX` clamp to NXm1 and write the wrong center value at their tile slot. Then a neighboring valid thread reads `tile[sy][sx+1]` which contains a clamped/duplicated value, not the true `i+1` value.

The fix: load tile values using actual (un-clamped) indices but bound to NX-1, and ensure halo threads load the correct neighbor regardless of clamping. I'll use a simpler 2D YX threadgroup tile where every thread cooperatively loads its center plus a clean halo strategy: dedicated halo loads using global coordinates, all guarded properly. I'll also keep z-direction in registers (no benefit per single step but cleaner). Use 32x8 tile.

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
                      uint3 gid [[thread_position_in_grid]],
                      uint3 tid [[thread_position_in_threadgroup]],
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

    // Origin of this threadgroup's tile in global coords.
    const uint i0 = tgid.x * TX;
    const uint j0 = tgid.y * TY;

    // Cooperative load of (TY+2) x (TX+2) tile from u_curr at slice k
    // (clamped to [0, NZm1] for safety; boundary threads won't use it).
    const uint kc = min(k, NZm1);
    const uint lid = ly * TX + lx;          // 0 .. TX*TY-1
    const uint total = TX * TY;             // threads in group
    const uint tile_n = SX * SY;            // cells in tile

    for (uint t = lid; t < tile_n; t += total) {
        uint sxi = t % SX;
        uint syi = t / SX;
        // Global coords for this tile cell: (i0 - 1 + sxi, j0 - 1 + syi).
        int gi = (int)i0 + (int)sxi - 1;
        int gj = (int)j0 + (int)syi - 1;
        gi = clamp(gi, 0, (int)NXm1);
        gj = clamp(gj, 0, (int)NYm1);
        tile[syi][sxi] = u_curr[(kc * NY + (uint)gj) * NX + (uint)gi];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (i >= NX || j >= NY || k >= NZ) return;

    const uint idx = (k * NY + j) * NX + i;

    if (i == 0u || j == 0u || k == 0u || i == NXm1 || j == NYm1 || k == NZm1) {
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