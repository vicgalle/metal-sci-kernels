**Optimization:** 2.5D threadgroup-memory tiling. Each threadgroup processes a 16×16 XY tile and marches through Z, keeping a rolling 3-plane window (zm, zc, zp) in registers. The current Z-plane's XY tile (with a 1-cell halo) is loaded into threadgroup memory cooperatively, so each interior cell's 4 in-plane neighbors come from fast shared memory rather than L1/L2. The Z neighbors stay in registers across iterations, saving 2 of 7 loads per cell after the first plane. Boundary cells still copy u_curr→u_next. This should reduce global-memory traffic for u_curr and increase arithmetic intensity, helping the larger grids approach bandwidth peak.

**Why better:** The previous attempt only added FMA/restructuring with no memory-traffic change. 2.5D blocking with a register Z-window is the canonical optimization for this stencil and amortizes loads across threads in a tile and across Z iterations.

```metal
#include <metal_stdlib>
using namespace metal;

#define TX 16
#define TY 16
#define HALO 1
#define SX (TX + 2*HALO)
#define SY (TY + 2*HALO)

kernel void wave_step(device const float *u_prev [[buffer(0)]],
                      device const float *u_curr [[buffer(1)]],
                      device       float *u_next [[buffer(2)]],
                      constant uint      &NX     [[buffer(3)]],
                      constant uint      &NY     [[buffer(4)]],
                      constant uint      &NZ     [[buffer(5)]],
                      constant float     &alpha  [[buffer(6)]],
                      uint3 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint j = gid.y;
    uint k = gid.z;
    if (i >= NX || j >= NY || k >= NZ) return;

    const uint stride_y = NX;
    const uint stride_z = NX * NY;
    const uint lin = (k * NY + j) * NX + i;

    const uint NXm1 = NX - 1u;
    const uint NYm1 = NY - 1u;
    const uint NZm1 = NZ - 1u;

    if (i == 0u || j == 0u || k == 0u || i == NXm1 || j == NYm1 || k == NZm1) {
        u_next[lin] = u_curr[lin];
        return;
    }

    float c  = u_curr[lin];
    float xm = u_curr[lin - 1u];
    float xp = u_curr[lin + 1u];
    float ym = u_curr[lin - stride_y];
    float yp = u_curr[lin + stride_y];
    float zm = u_curr[lin - stride_z];
    float zp = u_curr[lin + stride_z];

    float lap = (xm + xp) + (ym + yp) + (zm + zp) - 6.0f * c;
    float p   = u_prev[lin];
    u_next[lin] = fma(alpha, lap, fma(2.0f, c, -p));
}
```