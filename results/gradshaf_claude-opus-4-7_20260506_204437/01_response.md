**Optimization:** (1) Use simdgroup-based reduction with `simd_max` for the axis reduce, eliminating most threadgroup memory traffic and barriers. (2) For the stencil step, use threadgroup-memory tiling with halo cells: each 16×16 tile cooperatively loads an 18×N region (where N is the TG height + 2) into shared memory, then each thread reads its 5-point stencil from fast threadgroup memory instead of 5 device reads. This converts the memory-bound stencil from ~5 device loads per cell to ~1 amortized load per cell, dramatically reducing DRAM traffic. The tile dimensions exactly match the dispatched TG dims (16×TG_H), so no overlap. We support TG_H ∈ {16,8,4,2,1} via `threads_per_threadgroup`.

**Why faster:** The baseline issues 5 redundant device loads per cell; tiling shares loads across the TG. The reduction now uses 1 barrier + simd ops vs log2(256) barriers.

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
    threadgroup float simd_partials[32];

    float local_max = -INFINITY;
    uint NR_int = NR - 2u;
    uint NZ_int = NZ - 2u;
    uint total  = NR_int * NZ_int;

    // Strided sweep over interior cells.
    for (uint k = tid; k < total; k += tgsize) {
        uint i_int = k % NR_int;
        uint j_int = k / NR_int;
        float v = psi[(j_int + 1u) * NR + (i_int + 1u)];
        local_max = max(local_max, v);
    }

    // Simdgroup reduction.
    float sg_max = simd_max(local_max);
    uint sg_lane = tid & 31u;
    uint sg_id   = tid >> 5;
    if (sg_lane == 0u) {
        simd_partials[sg_id] = sg_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction by first simdgroup.
    if (sg_id == 0u) {
        uint num_sgs = (tgsize + 31u) >> 5;
        float v = (sg_lane < num_sgs) ? simd_partials[sg_lane] : -INFINITY;
        v = simd_max(v);
        if (sg_lane == 0u) {
            psi_axis[0] = v;
        }
    }
}

// Tile dims: 16 wide x up to 16 tall, plus halo of 1 on each side -> 18 x 18 max.
#define TILE_W 16
#define TILE_H 16
#define TILE_W_H (TILE_W + 2)
#define TILE_H_H (TILE_H + 2)

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
                          uint2 tgid [[threadgroup_position_in_grid]],
                          uint2 tgsz [[threads_per_threadgroup]]) {
    threadgroup float tile[TILE_H_H][TILE_W_H];

    uint i  = gid.x;
    uint j  = gid.y;
    uint lx = lid.x;          // 0..tg_w-1   (tg_w == 16)
    uint ly = lid.y;          // 0..tg_h-1
    uint tg_w = tgsz.x;       // 16
    uint tg_h = tgsz.y;       // 1,2,4,8,16

    // Tile origin in global coords (top-left of the interior cell block).
    int tile_i0 = int(tgid.x * tg_w);   // global i for lx=0
    int tile_j0 = int(tgid.y * tg_h);   // global j for ly=0

    // Cooperative load of (tg_w+2) x (tg_h+2) halo region into tile[ly+0..tg_h+1][lx+0..tg_w+1].
    // Each thread loads multiple elements as needed.
    uint halo_w = tg_w + 2u;
    uint halo_h = tg_h + 2u;
    uint flat   = ly * tg_w + lx;
    uint nthr   = tg_w * tg_h;
    uint halo_count = halo_w * halo_h;

    for (uint idx = flat; idx < halo_count; idx += nthr) {
        uint hx = idx % halo_w;
        uint hy = idx / halo_w;
        int gi = tile_i0 + int(hx) - 1;
        int gj = tile_j0 + int(hy) - 1;
        float v = 0.0f;
        if (gi >= 0 && gi < int(NR) && gj >= 0 && gj < int(NZ)) {
            v = psi_in[gj * int(NR) + gi];
        }
        tile[hy][hx] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (i >= NR || j >= NZ) return;

    if (i == 0u || j == 0u || i == NR - 1u || j == NZ - 1u) {
        psi_out[j * NR + i] = tile[ly + 1u][lx + 1u];
        return;
    }

    float R         = Rmin + float(i) * dR;
    float inv_dR2   = 1.0f / (dR * dR);
    float inv_dZ2   = 1.0f / (dZ * dZ);
    float h_inv_RdR = 0.5f / (R * dR);
    float a_W = inv_dR2 + h_inv_RdR;
    float a_E = inv_dR2 - h_inv_RdR;
    float a_N = inv_dZ2;
    float a_S = inv_dZ2;
    float a_C = -2.0f * inv_dR2 - 2.0f * inv_dZ2;

    float psi_C = tile[ly + 1u][lx + 1u];
    float psi_W = tile[ly + 1u][lx];
    float psi_E = tile[ly + 1u][lx + 2u];
    float psi_N = tile[ly + 2u][lx + 1u];
    float psi_S = tile[ly][lx + 1u];

    float ax = psi_axis[0];
    float psi_norm = psi_C / ax;
    float J = 0.0f;
    if (psi_norm > 0.0f && psi_norm < 1.0f) {
        J = R * p_axis * 4.0f * psi_norm * (1.0f - psi_norm);
    }
    float rhs = -mu0 * R * J;

    float delta_psi = a_W * psi_W + a_E * psi_E
                    + a_N * psi_N + a_S * psi_S
                    + a_C * psi_C;
    float r = rhs - delta_psi;
    psi_out[j * NR + i] = psi_C + omega * r / a_C;
}
```