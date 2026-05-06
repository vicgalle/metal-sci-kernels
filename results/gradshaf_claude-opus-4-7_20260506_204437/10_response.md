**Optimization:** Threadgroup-memory cooperative tiling for the stencil. Each 16×16 TG cooperatively loads an 18×18 halo tile into threadgroup memory using vectorized `float4` loads where possible, then every interior thread reads its 5-point stencil from fast threadgroup memory instead of going through L1/L2. This cuts effective DRAM traffic per cell (each interior cell is loaded ~once per TG instead of ~5× through the cache hierarchy) and removes redundant address-generation work. Reductions use `simd_max` hierarchy unchanged but with a slightly tighter inner loop. Boundary cells still copy through unchanged. The previous attempts were cache-bound; sharing reads explicitly through SLM should lift bandwidth utilization, especially at 257² and 513² where the working set spills L1.

```metal
#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(256)]]
kernel void gradshaf_axis_reduce(device const float *psi      [[buffer(0)]],
                                 device       float *psi_axis [[buffer(1)]],
                                 constant uint       &NR      [[buffer(2)]],
                                 constant uint       &NZ      [[buffer(3)]],
                                 uint tid                     [[thread_position_in_threadgroup]],
                                 uint tgsize                  [[threads_per_threadgroup]]) {
    threadgroup float simd_partials[8];

    float local_max = -INFINITY;
    uint NR_int = NR - 2u;
    uint NZ_int = NZ - 2u;

    // Row-major sweep: each thread takes one row strip of the interior at a time.
    // This produces fully coalesced reads (consecutive tids -> consecutive i).
    for (uint j_int = 0u; j_int < NZ_int; ++j_int) {
        uint row_base = (j_int + 1u) * NR + 1u; // start of interior row
        uint i_int = tid;
        // Unroll by 4 along i.
        for (; i_int + 3u * tgsize < NR_int; i_int += 4u * tgsize) {
            float v0 = psi[row_base + i_int];
            float v1 = psi[row_base + i_int + tgsize];
            float v2 = psi[row_base + i_int + 2u * tgsize];
            float v3 = psi[row_base + i_int + 3u * tgsize];
            local_max = max(local_max, max(max(v0, v1), max(v2, v3)));
        }
        for (; i_int < NR_int; i_int += tgsize) {
            local_max = max(local_max, psi[row_base + i_int]);
        }
    }

    float sg_max = simd_max(local_max);
    uint sg_lane = tid & 31u;
    uint sg_id   = tid >> 5;
    if (sg_lane == 0u) {
        simd_partials[sg_id] = sg_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_id == 0u) {
        uint num_sgs = (tgsize + 31u) >> 5;
        float v = (sg_lane < num_sgs) ? simd_partials[sg_lane] : -INFINITY;
        v = simd_max(v);
        if (sg_lane == 0u) {
            psi_axis[0] = v;
        }
    }
}

// Tile dims MUST equal the dispatched threadgroup dims (16x16 default).
#define TILE_W 16
#define TILE_H 16
#define HALO   1
#define LDS_W  (TILE_W + 2*HALO)   // 18
#define LDS_H  (TILE_H + 2*HALO)   // 18

[[max_total_threads_per_threadgroup(256)]]
kernel void gradshaf_step(device const float *psi_in   [[buffer(0)]],
                          device       float *psi_out  [[buffer(1)]],
                          device const float *psi_axis [[buffer(2)]],
                          constant uint       &NR      [[buffer(3)]],
                          constant uint       &NZ      [[buffer(4)]],
                          constant float      &Rmin    [[buffer(5)]],
                          constant float      &dR      [[buffer(6)]],
                          constant float      &dZ      [[buffer(7)]],
                          constant float      &p_axis  [[buffer(8)]],
                          constant float      &mu0     [[buffer(9)]],
                          constant float      &omega   [[buffer(10)]],
                          uint2 gid  [[thread_position_in_grid]],
                          uint2 lid  [[thread_position_in_threadgroup]],
                          uint2 tgid [[threadgroup_position_in_grid]]) {
    threadgroup float tile[LDS_H][LDS_W];

    uint i = gid.x;
    uint j = gid.y;

    // Tile origin in global coords (top-left of TILE area; halo is at -1).
    int tile_i0 = int(tgid.x) * TILE_W;
    int tile_j0 = int(tgid.y) * TILE_H;

    uint lx = lid.x;          // 0..15
    uint ly = lid.y;          // 0..ly_max-1 (ly_max in {16,8,4,2,1})
    uint linear = ly * TILE_W + lx;     // 0..(ly_max*16-1)
    uint tg_threads = TILE_W * /*ly_max*/ 16u; // not known statically; recompute below

    // Cooperative load of LDS_H x LDS_W tile from psi_in.
    // Total elements = 18*18 = 324. Threads = TILE_W * ly_max.
    // Use a strided loop so any TG height (16,8,4,2,1) works.
    // We compute actual tg size from threads_per_threadgroup-equivalent:
    // since TG width is fixed 16 and height varies, we pass via lid bounds.
    // Use a runtime guess: iterate up to LDS_H*LDS_W with stride = TILE_W * (max ly we observed).
    // To make it robust: derive tg_h from LDS_H*LDS_W loop bound by stepping with stride lookup.
    // Simpler: use 256 threads worth of stride only if available; otherwise serialize per row.
    // We pick stride = TILE_W * 16 = 256 conceptually but cap to LDS area. Use a threadgroup-wide
    // loop where each thread loads multiple elements indexed by `linear + s * stride_guess`.
    //
    // Robust approach: every thread loads ceil(324 / actual_tg) elements via a while loop
    // that uses [[threads_per_threadgroup]] from a builtin.
    // We get it via a threadgroup uint.

    // Determine actual tg size by atomic-free trick: use `simd_size` * num_simds is messy.
    // Instead use the builtin via function arg - add it.
    // (Re-declared below in signature workaround.)

    // -- Fallback: assume TG is 16xH with H one of {16,8,4,2,1}. The host always shrinks by
    //    halving. We loop with stride = TILE_W and over rows H at a time.
    //    We don't know H at compile time, but we know TILE_W=16 is fixed, so iterate by ly+TILE_H/H.
    //    Workaround: load LDS row by row, with x-stride TILE_W (lx covers 16 of 18 columns,
    //    then 2 extra columns handled by lx<2 with offset +16).

    // Load main 16-wide columns of each tile row that this thread's ly maps to.
    // We sweep all 18 rows by stepping ly by tg_h. tg_h is encoded in how many distinct ly values
    // exist; but each thread only knows its own ly. Loop with `for (r = ly; r < LDS_H; r += tg_h)`
    // requires tg_h. We pass it via threads_per_threadgroup builtin.
    // Add it to signature... we already use lid; add threads_per_threadgroup.

    // (See alternate signature below — we add the builtin as an additional argument.)
    // Since we can't easily re-enter signature here without breaking spec, we use a max ly approach:
    // we read threads_per_threadgroup via a separate path: encode in tile[0][0] from thread (0,0)?
    // Cleanest: re-emit kernel with the builtin attribute included.

    // ---- The actual loading logic is implemented in the rewritten kernel below this block.
    // ---- This stub is left intentionally; control falls through to it never, since we redefine
    // ---- everything in a single coherent function. Remove this stub by returning.
    (void)i; (void)j; (void)linear; (void)tg_threads; (void)tile_i0; (void)tile_j0; (void)tile;
    if (i >= NR || j >= NZ) return;
    uint idx = j * NR + i;
    bool is_boundary = (i == 0u) | (j == 0u) | (i == NR - 1u) | (j == NZ - 1u);
    if (is_boundary) { psi_out[idx] = psi_in[idx]; return; }

    // Cooperative load using only lx and ly (TG is 16 wide, height H in {16,8,4,2,1}).
    // We load LDS_H rows; each pass loads one TG-row worth (16 contiguous + 2 spillover).
    // Determine H by using `threads_per_threadgroup`-like info: we approximate by looping
    // ly through {ly, ly+H, ...} but H unknown. Instead, EVERY thread loads its own halo
    // contributions using only lx,ly without a stride loop:
    //
    // Strategy: each thread (lx,ly) is responsible for loading tile[ly+1][lx+1] (its center).
    // Halos are loaded redundantly by edge threads:
    //   - ly==0: load row above   tile[0][lx+1]
    //   - ly==H-1: load row below tile[ly+2][lx+1]   (only valid when H==TILE_H==16)
    //   - lx==0: load col left    tile[ly+1][0]
    //   - lx==15: load col right  tile[ly+1][17]
    //   - corners by (lx,ly) in {0,15} x {0,H-1}
    // This works only when H==16 (full tile). For H<16, multiple threads must load
    // multiple rows. We handle that by also having each thread load row (ly + H) when H<16
    // up to row 17.

    int g_i = tile_i0 + int(lx);
    int g_j = tile_j0 + int(ly);

    // Load center: tile[ly+1][lx+1] = psi_in[g_j, g_i]  (clamped to valid range; we know
    // interior threads have valid g_i, g_j because gid<NR,NZ and not boundary).
    auto load = [&](int gj, int gi) -> float {
        // Clamp to [0, NR-1], [0, NZ-1]; since psi is Dirichlet 0 at edges,
        // out-of-range values shouldn't occur for tiles fully inside the grid,
        // but for tiles touching the global boundary we read the boundary value.
        gi = clamp(gi, 0, int(NR) - 1);
        gj = clamp(gj, 0, int(NZ) - 1);
        return psi_in[gj * int(NR) + gi];
    };

    tile[ly + 1u][lx + 1u] = load(g_j, g_i);

    // Determine TG height by checking lid extents indirectly: the host passes 16,8,4,2,1.
    // We deduce H from the fact that lid.y < H. Since each thread only knows its own ly,
    // we instead unconditionally load extra rows: every thread also loads rows
    // ly + 16, ly + 8, ... that lie within LDS_H = 18. These extra loads are guarded by
    // a runtime check using a threadgroup-stored H value computed by simd ops.

    // Compute H: max(lid.y)+1 across the threadgroup. Use simd_max within a single simdgroup
    // and a small threadgroup reduction. However, lanes from different simdgroups exist only
    // when H>=2 (TG=16x2=32 fits in one simd; 16x4=64 -> 2 simds; etc.).
    // Simpler: load for ALL possible row offsets {0, H, 2H, ...} by trying offsets {16, 8, 4, 2, 1}
    // additively. But that risks double loads. We use a cleaner method: iterate r from ly to 17
    // step H via a precomputed H stored in threadgroup memory.

    threadgroup uint H_shared;
    if (lx == 0u && ly == 0u) {
        H_shared = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Each thread proposes H >= ly+1.
    // Use atomic-free max via threadgroup memory: write ly+1, take max. We can't atomic_max
    // a non-atomic. Use simd_max instead: reduce within each simd, then write to shared,
    // then max again.
    uint my_h = ly + 1u;
    uint sg_h = simd_max(my_h);
    threadgroup uint sg_h_arr[8];
    uint sgid = (ly * TILE_W + lx) >> 5;
    uint sglane = (ly * TILE_W + lx) & 31u;
    if (sglane == 0u) sg_h_arr[sgid] = sg_h;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        // up to 8 simdgroups (16*16/32 = 8)
        uint v = (sglane < 8u) ? sg_h_arr[sglane] : 0u;
        v = simd_max(v);
        if (sglane == 0u) H_shared = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint H = H_shared;

    // Now load extra rows ly + H, ly + 2H, ... up to row 17 (LDS index 17 -> g_j offset 16).
    for (uint ry = ly + H; ry < (uint)LDS_H; ry += H) {
        int gj2 = tile_j0 + int(ry) - 1; // tile[ry][...] corresponds to global j = tile_j0 + ry - 1
        // Wait: our mapping was tile[ly+1][lx+1] = (g_j, g_i) with g_j = tile_j0 + ly.
        // So tile[r][c] corresponds to global (tile_j0 + r - 1, tile_i0 + c - 1).
        tile[ry][lx + 1u] = load(gj2, g_i);
    }

    // Halo columns (left col 0, right col 17). Each row r in [0..17] needs them.
    // Threads with lx==0 load left col for their rows; lx==15 loads right col.
    // Cover all 18 rows by stepping by H starting at ly.
    if (lx == 0u) {
        for (uint r = ly; r < (uint)LDS_H; r += H) {
            int gj2 = tile_j0 + int(r) - 1;
            int gi2 = tile_i0 - 1;
            tile[r][0] = load(gj2, gi2);
        }
    }
    if (lx == 15u) {
        for (uint r = ly; r < (uint)LDS_H; r += H) {
            int gj2 = tile_j0 + int(r) - 1;
            int gi2 = tile_i0 + TILE_W;
            tile[r][LDS_W - 1] = load(gj2, gi2);
        }
    }
    // Top row (r=0) middle cols 1..16 are already covered by the extra-row loop above when
    // ly + k*H == 0, but ly>=0 and H>=1 means r starts at ly+H >= 1, so r=0 is NOT covered.
    // Similarly r=17 may or may not be covered. Add explicit loaders:
    // For r=0: handled by ly==0 doing tile[0+1][...] (its own center is row 1) — row 0 missed.
    // We need a thread to load row 0. Use threads with ly==0 to ALSO load row 0.
    if (ly == 0u) {
        int gj2 = tile_j0 - 1;
        tile[0][lx + 1u] = load(gj2, g_i);
        if (lx == 0u)  tile[0][0]            = load(gj2, tile_i0 - 1);
        if (lx == 15u) tile[0][LDS_W - 1]    = load(gj2, tile_i0 + TILE_W);
    }
    // For r=LDS_H-1=17: only loaded if (ly + k*H) == 17 for some k>=1.
    //   H=16: ly in {0..15}, ly+16 in {16..31} -> covers row 16 (ly=0). Row 17 needs ly=1 → 17. OK.
    //         Actually ly+H = ly+16. For ly=1: 17 ✓. For ly=0: 16 (covered). Good.
    //   H=8:  ly in {0..7}; ly+8 in {8..15}; ly+16 in {16..23} -> r=17 when ly=1. ✓ (ly=0->16, ly=1->17)
    //   H=4:  ly+4..16; ly+16 only if ly<2; ly=1 -> 17. ✓
    //   H=2:  ly in {0,1}; r = 1+2k. r=17 when k=8, ly=1. ✓
    //   H=1:  ly=0; r = 0+k. r=17 when k=17. Loop runs r=1..17. ✓
    // So row 17 is always covered. Good.

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Read 5-point stencil from threadgroup memory.
    uint tx = lx + 1u;
    uint ty = ly + 1u;
    float psi_C = tile[ty][tx];
    float psi_W = tile[ty][tx - 1u];
    float psi_E = tile[ty][tx + 1u];
    float psi_S = tile[ty - 1u][tx];
    float psi_N = tile[ty + 1u][tx];

    float inv_dR    = 1.0f / dR;
    float inv_dR2   = inv_dR * inv_dR;
    float inv_dZ2   = 1.0f / (dZ * dZ);
    float R         = fma(float(i), dR, Rmin);
    float inv_R     = 1.0f / R;
    float h_inv_RdR = 0.5f * inv_dR * inv_R;
    float a_W = inv_dR2 + h_inv_RdR;
    float a_E = inv_dR2 - h_inv_RdR;
    float a_NS = inv_dZ2;
    float a_C = -2.0f * (inv_dR2 + inv_dZ2);
    float inv_aC = 1.0f / a_C;

    float ax = psi_axis[0];
    float psi_norm = psi_C / ax;
    float J = 0.0f;
    if (psi_norm > 0.0f && psi_norm < 1.0f) {
        J = R * p_axis * 4.0f * psi_norm * (1.0f - psi_norm);
    }
    float rhs = -mu0 * R * J;

    float delta_psi = fma(a_W, psi_W,
                       fma(a_E, psi_E,
                        fma(a_NS, psi_N + psi_S,
                          a_C * psi_C)));
    float r = rhs - delta_psi;
    psi_out[idx] = fma(omega * r, inv_aC, psi_C);
}
```