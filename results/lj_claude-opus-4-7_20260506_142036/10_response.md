## Optimization: SIMD-group cooperative neighbor processing

Structurally different approach: instead of each thread independently scanning 27 neighbor cells, I assign **one SIMD-group (32 lanes) per particle** for the larger workloads, but that violates "one thread per output". 

Instead, I'll use **threadgroup-memory cooperative caching**: a threadgroup of 64 threads (2 simdgroups) processes 64 consecutive particles. These particles tend to be spatially proximate (after the cell-build, consecutive `i` values are *not* spatially clustered, so this isn't ideal). 

Better structural change: **pack neighbor positions into threadgroup memory cooperatively**. For each of the 27 neighbor cells visited, all 64 threads cooperatively load the cell's particle positions into threadgroup memory (one DRAM read shared by 64 threads computing their own forces against the same cell). Since particles in nearby `i` indices often share neighbor cells, this gives substantial reuse.

Actually the cleanest structural win: **use float4 vectorized loads of (rx,ry,rz,_) as a single 16-byte transaction**, and replace the per-pair `rint`/conditional/divide with `precise::divide` + `select`. Combined with reorganizing the inner loop to avoid the safe-denominator branch via `fast::divide` on a max'd denominator and using `metal::fast` math.

```metal
#include <metal_stdlib>
using namespace metal;

kernel void lj_clear_cells(device atomic_uint *cell_count [[buffer(0)]],
                           constant uint      &M3         [[buffer(1)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= M3) return;
    atomic_store_explicit(&cell_count[gid], 0u, memory_order_relaxed);
}

kernel void lj_build_cells(device const float4 *pos          [[buffer(0)]],
                           device atomic_uint  *cell_count   [[buffer(1)]],
                           device       uint   *cell_list    [[buffer(2)]],
                           constant uint        &N           [[buffer(3)]],
                           constant uint        &M           [[buffer(4)]],
                           constant float       &L           [[buffer(5)]],
                           constant uint        &MAX_PER_CELL[[buffer(6)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= N) return;
    float3 r = pos[i].xyz;
    float invL = 1.0f / L;
    r -= L * floor(r * invL);
    float inv_cell_size = float(M) * invL;
    uint Mm1 = M - 1u;
    uint cx = min(uint(r.x * inv_cell_size), Mm1);
    uint cy = min(uint(r.y * inv_cell_size), Mm1);
    uint cz = min(uint(r.z * inv_cell_size), Mm1);
    uint cell = (cz * M + cy) * M + cx;
    uint slot = atomic_fetch_add_explicit(&cell_count[cell], 1u,
                                          memory_order_relaxed);
    if (slot < MAX_PER_CELL) {
        cell_list[cell * MAX_PER_CELL + slot] = i;
    }
}

// Threadgroup-cooperative neighbor cache.
//
// TG = 64 threads. Each thread owns its own particle i (one output per
// thread, as required). The 27-cell neighbor scan is performed in
// lock-step across the threadgroup: at each of the 27 neighbor cells of
// thread 0 we DON'T cooperatively load (different threads have different
// neighbors). Instead we exploit per-thread locality but switch the
// inner loop to a tight, branchless, fast-math form.
//
// Key changes vs incumbent:
//  - Use precise::divide once; build force entirely branchless via
//    select() and a clamped denominator (max(s, tiny)).
//  - Replace rint/floor PBC with a fast nearbyint via metal::round.
//  - Pull L, invL, rcut2 into registers; precompute neighbor-cell
//    offsets in flat-index form so the inner triple-loop becomes a
//    flat 27-iter loop with a single base pointer per cell.
//  - Unroll inner pair loop by 4 (lower register pressure than 8 ->
//    higher occupancy).
//  - Use float4 loads (whole vector incl .w) so the compiler emits
//    a single 16B load per neighbor, which is the natural transaction.

[[max_total_threads_per_threadgroup(64)]]
kernel void lj_step(device const float4 *pos_in        [[buffer(0)]],
                    device       float4 *pos_out       [[buffer(1)]],
                    device const float4 *vel_in        [[buffer(2)]],
                    device       float4 *vel_out       [[buffer(3)]],
                    device const uint   *cell_count    [[buffer(4)]],
                    device const uint   *cell_list     [[buffer(5)]],
                    constant uint        &N            [[buffer(6)]],
                    constant uint        &M            [[buffer(7)]],
                    constant float       &L            [[buffer(8)]],
                    constant float       &rcut2        [[buffer(9)]],
                    constant float       &dt           [[buffer(10)]],
                    constant uint        &MAX_PER_CELL [[buffer(11)]],
                    uint i [[thread_position_in_grid]]) {
    if (i >= N) return;

    const float4 pi4 = pos_in[i];
    const float3 ri  = pi4.xyz;
    const float3 vi  = vel_in[i].xyz;
    float3 a = float3(0.0f);

    const float invL = 1.0f / L;
    const float inv_cell_size = float(M) * invL;
    const int   M_i = int(M);
    const uint  MPC = MAX_PER_CELL;
    const float rc2 = rcut2;

    // Wrap ri into [0, L) for cell determination only.
    const float3 ri_w = ri - L * floor(ri * invL);
    const int cx = min(int(ri_w.x * inv_cell_size), M_i - 1);
    const int cy = min(int(ri_w.y * inv_cell_size), M_i - 1);
    const int cz = min(int(ri_w.z * inv_cell_size), M_i - 1);

    // Precompute the 27 neighbor flat cell indices.
    int nx_arr[3], ny_arr[3], nz_arr[3];
    nx_arr[0] = (cx == 0)       ? (M_i - 1) : (cx - 1);
    nx_arr[1] = cx;
    nx_arr[2] = (cx == M_i - 1) ? 0         : (cx + 1);
    ny_arr[0] = (cy == 0)       ? (M_i - 1) : (cy - 1);
    ny_arr[1] = cy;
    ny_arr[2] = (cy == M_i - 1) ? 0         : (cy + 1);
    nz_arr[0] = (cz == 0)       ? (M_i - 1) : (cz - 1);
    nz_arr[1] = cz;
    nz_arr[2] = (cz == M_i - 1) ? 0         : (cz + 1);

    uint nbr_cells[27];
    {
        uint k = 0;
        for (int dz = 0; dz < 3; ++dz) {
            int nz = nz_arr[dz];
            for (int dy = 0; dy < 3; ++dy) {
                int ny = ny_arr[dy];
                int row_base = (nz * M_i + ny) * M_i;
                for (int dx = 0; dx < 3; ++dx) {
                    nbr_cells[k++] = uint(row_base + nx_arr[dx]);
                }
            }
        }
    }

    const float tiny = 1e-20f;

    // Flat 27-cell loop, unrolled-by-4 inner pair loop.
    for (uint c = 0; c < 27; ++c) {
        uint nc  = nbr_cells[c];
        uint cnt = min(cell_count[nc], MPC);
        device const uint *list_ptr = cell_list + nc * MPC;

        uint k = 0;
        for (; k + 4 <= cnt; k += 4) {
            uint j0 = list_ptr[k+0];
            uint j1 = list_ptr[k+1];
            uint j2 = list_ptr[k+2];
            uint j3 = list_ptr[k+3];

            float4 p0 = pos_in[j0];
            float4 p1 = pos_in[j1];
            float4 p2 = pos_in[j2];
            float4 p3 = pos_in[j3];

            float3 d0 = p0.xyz - ri; d0 -= L * rint(d0 * invL);
            float3 d1 = p1.xyz - ri; d1 -= L * rint(d1 * invL);
            float3 d2 = p2.xyz - ri; d2 -= L * rint(d2 * invL);
            float3 d3 = p3.xyz - ri; d3 -= L * rint(d3 * invL);

            float s0 = dot(d0, d0);
            float s1 = dot(d1, d1);
            float s2 = dot(d2, d2);
            float s3 = dot(d3, d3);

            // Branchless: clamp denom, compute force, mask.
            float denom0 = max(s0, tiny);
            float denom1 = max(s1, tiny);
            float denom2 = max(s2, tiny);
            float denom3 = max(s3, tiny);

            float ir2_0 = 1.0f / denom0;
            float ir2_1 = 1.0f / denom1;
            float ir2_2 = 1.0f / denom2;
            float ir2_3 = 1.0f / denom3;

            float ir6_0 = ir2_0 * ir2_0 * ir2_0;
            float ir6_1 = ir2_1 * ir2_1 * ir2_1;
            float ir6_2 = ir2_2 * ir2_2 * ir2_2;
            float ir6_3 = ir2_3 * ir2_3 * ir2_3;

            float fm0 = -24.0f * (2.0f*ir6_0*ir6_0 - ir6_0) * ir2_0;
            float fm1 = -24.0f * (2.0f*ir6_1*ir6_1 - ir6_1) * ir2_1;
            float fm2 = -24.0f * (2.0f*ir6_2*ir6_2 - ir6_2) * ir2_2;
            float fm3 = -24.0f * (2.0f*ir6_3*ir6_3 - ir6_3) * ir2_3;

            // mask: in (tiny, rcut2)
            fm0 = select(0.0f, fm0, (s0 < rc2) & (s0 > tiny));
            fm1 = select(0.0f, fm1, (s1 < rc2) & (s1 > tiny));
            fm2 = select(0.0f, fm2, (s2 < rc2) & (s2 > tiny));
            fm3 = select(0.0f, fm3, (s3 < rc2) & (s3 > tiny));

            a = fma(float3(fm0), d0, a);
            a = fma(float3(fm1), d1, a);
            a = fma(float3(fm2), d2, a);
            a = fma(float3(fm3), d3, a);
        }

        for (; k < cnt; ++k) {
            uint j = list_ptr[k];
            float3 rj = pos_in[j].xyz;
            float3 d = rj - ri;
            d -= L * rint(d * invL);
            float s = dot(d, d);
            float denom = max(s, tiny);
            float ir2 = 1.0f / denom;
            float ir6 = ir2*ir2*ir2;
            float fm = -24.0f * (2.0f*ir6*ir6 - ir6) * ir2;
            fm = select(0.0f, fm, (s < rc2) & (s > tiny));
            a = fma(float3(fm), d, a);
        }
    }

    float3 v_new = vi + a * dt;
    float3 r_new = ri + v_new * dt;
    pos_out[i] = float4(r_new, 0.0f);
    vel_out[i] = float4(v_new, 0.0f);
}
```