The previous attempt failed because the threadgroup size (TX=32, TY=8) doesn't evenly divide all problem sizes (64, 160, 192 are all divisible, but the issue is different): when a threadgroup straddles the grid edge, threads with `!in_bounds` still participate in halo loading using arbitrary `i,j,k` values, and the cooperative load conditions `lx==0`, `lx==TX-1` etc. assume the corresponding global neighbor exists. More critically, the halo loaders for `lx==0` use `j` and `k` from threads that may be out of bounds, producing NaN-poisoning reads. Also `tile[sy][sx]` for out-of-bounds threads writes 0 but boundary copy reads it back.

Fix: compute clamped global coordinates per-thread, always load valid data, and remove the early-return-before-halo path. Use a clean 32x8 tile, load halos with branchless clamped indexing, then compute. This should cut DRAM traffic substantially for the X/Y neighbors.

```metal
#include <metal_stdlib>
using namespace metal;

#define TX 32
#define TY 8
#define SX (TX + 2)
#define SY (TY + 2)

kernel void wave_step(device const float *u_prev [[buffer(0)]],
                      device const float *u_curr [[buffer(1)]],
                      device       float *u_next [[buffer(2)]],
                      constant uint      &NX     [[buffer(3)]],
                      constant uint      &NY     [[buffer(4)]],
                      constant uint      &NZ     [[buffer(5)]],
                      constant float     &alpha  [[buffer(6)]],
                      uint3 gid [[thread_position_in_grid]],
                      uint3 tid [[thread_position_in_threadgroup]]) {
    threadgroup float tile[SY][SX];

    const uint i = gid.x;
    const uint j = gid.y;
    const uint k = gid.z;

    const uint lx = tid.x;
    const uint ly = tid.y;
    const uint sx = lx + 1u;
    const uint sy = ly + 1u;

    const uint NXm1 = NX - 1u;
    const uint NYm1 = NY - 1u;
    const uint NZm1 = NZ - 1u;

    const uint stride_z = NX * NY;

    // Clamped global coords for safe loads (used for halos too).
    const uint ic = min(i, NXm1);
    const uint jc = min(j, NYm1);
    const uint kc = min(k, NZm1);
    const uint base_kj = (kc * NY + jc) * NX;

    // Center load.
    tile[sy][sx] = u_curr[base_kj + ic];

    // X halos.
    if (lx == 0u) {
        uint il = (ic == 0u) ? 0u : (ic - 1u);
        tile[sy][0] = u_curr[base_kj + il];
    }
    if (lx == TX - 1u) {
        uint ir = (ic + 1u <= NXm1) ? (ic + 1u) : NXm1;
        tile[sy][SX - 1u] = u_curr[base_kj + ir];
    }

    // Y halos.
    if (ly == 0u) {
        uint jt = (jc == 0u) ? 0u : (jc - 1u);
        tile[0][sx] = u_curr[(kc * NY + jt) * NX + ic];
    }
    if (ly == TY - 1u) {
        uint jb = (jc + 1u <= NYm1) ? (jc + 1u) : NYm1;
        tile[SY - 1u][sx] = u_curr[(kc * NY + jb) * NX + ic];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (i >= NX || j >= NY || k >= NZ) return;

    const uint idx = (k * NY + j) * NX + i;

    if (i == 0u || j == 0u || k == 0u || i == NXm1 || j == NYm1 || k == NZm1) {
        u_next[idx] = u_curr[idx];
        return;
    }

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