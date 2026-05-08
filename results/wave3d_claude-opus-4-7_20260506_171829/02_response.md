The previous attempt failed because of a threadgroup memory hazard: boundary threads returned early *after* the barrier, but threads at the tile edges (e.g., `lx == TX-1`) were responsible for loading the right halo. When `i == NX-1` happens to fall inside a tile (not at `lx == TX-1`), that thread loaded the right halo and then the barrier was fine — but the actual NaN bug is that when a tile's right edge thread is at a domain boundary it returns, while neighbors in the same threadgroup still expect the halo that thread should have loaded *before* the barrier. Actually, the load happens before the barrier so that's OK... the real issue: when `i == NX-1` triggers the right-halo load, but the thread itself also satisfies `lx == TX-1`, both branches fire safely. NaN likely came from not loading center for out-of-range threads when NX isn't a multiple of TX — those threads return before storing to tile, leaving uninitialized values used by neighbors.

Fix: do all threadgroup loads unconditionally with clamped indices BEFORE any early return, then compute. Use a 32×8 XY tile with Z reads from global memory (Z neighbors are far apart anyway). Use `fma` for the final update.

```metal
#include <metal_stdlib>
using namespace metal;

#define TX 32u
#define TY 8u

kernel void wave_step(device const float *u_prev [[buffer(0)]],
                      device const float *u_curr [[buffer(1)]],
                      device       float *u_next [[buffer(2)]],
                      constant uint      &NX     [[buffer(3)]],
                      constant uint      &NY     [[buffer(4)]],
                      constant uint      &NZ     [[buffer(5)]],
                      constant float     &alpha  [[buffer(6)]],
                      uint3 gid [[thread_position_in_grid]],
                      uint3 lid [[thread_position_in_threadgroup]],
                      uint3 tgid [[threadgroup_position_in_grid]]) {
    threadgroup float tile[TY + 2u][TX + 2u];

    const uint lx = lid.x;
    const uint ly = lid.y;

    // Tile origin in global coords
    const uint i0 = tgid.x * TX;
    const uint j0 = tgid.y * TY;
    const uint k  = gid.z;

    const uint i = i0 + lx;
    const uint j = j0 + ly;

    const uint stride_z = NX * NY;

    // Clamp k for safe slab loads
    const uint kc = (k < NZ) ? k : (NZ - 1u);

    // Helper lambda would be nice; inline instead.
    // Load center: clamp (i,j) to valid range so all tile cells are defined.
    {
        uint ii = (i < NX) ? i : (NX - 1u);
        uint jj = (j < NY) ? j : (NY - 1u);
        tile[ly + 1u][lx + 1u] = u_curr[(kc * NY + jj) * NX + ii];
    }

    // Left/right halos (lx == 0 loads left, lx == TX-1 loads right)
    if (lx == 0u) {
        uint ii = (i == 0u) ? 0u : (i - 1u);
        uint jj = (j < NY) ? j : (NY - 1u);
        tile[ly + 1u][0] = u_curr[(kc * NY + jj) * NX + ii];
    }
    if (lx == TX - 1u) {
        uint ii_raw = i + 1u;
        uint ii = (ii_raw < NX) ? ii_raw : (NX - 1u);
        uint jj = (j < NY) ? j : (NY - 1u);
        tile[ly + 1u][TX + 1u] = u_curr[(kc * NY + jj) * NX + ii];
    }
    // Top/bottom halos
    if (ly == 0u) {
        uint jj = (j == 0u) ? 0u : (j - 1u);
        uint ii = (i < NX) ? i : (NX - 1u);
        tile[0][lx + 1u] = u_curr[(kc * NY + jj) * NX + ii];
    }
    if (ly == TY - 1u) {
        uint jj_raw = j + 1u;
        uint jj = (jj_raw < NY) ? jj_raw : (NY - 1u);
        uint ii = (i < NX) ? i : (NX - 1u);
        tile[ly + 2u][lx + 1u] = u_curr[(kc * NY + jj) * NX + ii];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Out-of-range threads (when NX/NY not multiple of TX/TY) bail now.
    if (i >= NX || j >= NY || k >= NZ) return;

    const uint idx = (k * NY + j) * NX + i;
    const float c_val = tile[ly + 1u][lx + 1u];

    // Dirichlet boundary
    if (i == 0u || j == 0u || k == 0u
        || i == NX - 1u || j == NY - 1u || k == NZ - 1u) {
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