**Optimization:** The previous attempt produced NaN, likely because `kc` was clamped to `NZm1` for out-of-range z-threads, but the early-return after the barrier still skipped writes correctly. The issue is likely that `tgid.z` ranged beyond NZ when NZ%1 != 0... actually with `threadsPerGrid=(NX,NY,NZ)` and threadgroup `(TX,TY,1)`, `gid.z` always equals `k` and is in-bounds. The NaN likely came from `idx - stride_z` when the boundary check missed something — wait, the boundary check guards k==0 and k==NZ-1, so that's safe. Most likely cause: when NX or NY is not a multiple of TX/TY, the cooperative tile load is fine, but threads with `i>=NX` skipped the barrier in some path... actually all threads hit the barrier before the early-out. Let me use a simpler, robust approach: all threads stay alive through the barrier, use threadgroup tiling for X/Y stencil neighbors with proper bounds clamping, and read Z neighbors directly. I'll dispatch with `(TX,TY,1)` threadgroups and ensure no thread early-exits before the barrier.

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

    // Clamp k for the cooperative load so all threads (including those
    // with k >= NZ) read a valid in-bounds slice. The actual write is
    // gated below.
    const uint kc = (k < NZ) ? k : NZm1;

    const uint lid = ly * TX + lx;
    const uint total = TX * TY;
    const uint tile_n = SX * SY;

    // Cooperative load of the (TY+2) x (TX+2) halo tile from u_curr at
    // slice kc. ALL threads in the threadgroup participate; none has
    // returned yet, so the barrier below is safe.
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

    // Now safe to skip out-of-range threads.
    if (i >= NX || j >= NY || k >= NZ) return;

    const uint idx = (k * NY + j) * NX + i;

    // Dirichlet boundary: copy u_curr through unchanged.
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