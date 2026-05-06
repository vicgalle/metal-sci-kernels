**Optimization:** Switch the step kernel to threadgroup-memory tiling for the 5-point stencil. With a 16×16 dispatched TG, I load an 18×18 halo tile cooperatively into shared memory once, then every interior thread reads its 5 neighbors from threadgroup memory instead of device memory. This collapses the effective DRAM read traffic from ~5 loads/cell (cache-dependent) to ~1 load/cell amortized, which is the right move for a BW-bound kernel running at only ~4% of roofline. Halo loads use clamped indices so out-of-range threads still safely fetch in-bounds data (boundary cells then take the copy-through path and write `psi_in` unchanged). The reduce kernel is unchanged in spirit but cleaned up.

(Note: the system reminder about keybindings-help is not relevant to this Metal kernel task.)

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
    uint total = NR_int * NZ_int;

    for (uint k = tid; k < total; k += tgsize) {
        uint j_int = k / NR_int;
        uint i_int = k - j_int * NR_int;
        float v = psi[(j_int + 1u) * NR + (i_int + 1u)];
        local_max = max(local_max, v);
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

// Tile dims must equal the dispatched TG dims. Host always uses tg_w = 16 and
// tg_h ∈ {16,8,4,2,1}, capped by max_total_threads_per_threadgroup. With the
// 256 cap we get the default 16×16 tile + 1-cell halo => 18×18 = 324 floats.
#define TILE_W 16
#define TILE_H 16
#define HALO_W (TILE_W + 2)
#define HALO_H (TILE_H + 2)

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
                          uint2 tgid [[threadgroup_position_in_grid]],
                          uint2 ltid [[thread_position_in_threadgroup]],
                          uint2 tgsz [[threads_per_threadgroup]]) {
    threadgroup float tile[HALO_H][HALO_W];

    // Origin of this tile in global coords (must use tgsz, since tg_h may be < TILE_H)
    int tile_i0 = int(tgid.x) * int(tgsz.x);
    int tile_j0 = int(tgid.y) * int(tgsz.y);

    uint lx = ltid.x;
    uint ly = ltid.y;
    uint lin = ly * tgsz.x + lx;
    uint nthreads = tgsz.x * tgsz.y;

    int NRi = int(NR);
    int NZi = int(NZ);

    // Cooperative halo load: fill (tgsz.y + 2) x (tgsz.x + 2) region.
    uint halo_w = tgsz.x + 2u;
    uint halo_h = tgsz.y + 2u;
    uint halo_total = halo_w * halo_h;

    for (uint k = lin; k < halo_total; k += nthreads) {
        uint hy = k / halo_w;
        uint hx = k - hy * halo_w;
        int gx = tile_i0 + int(hx) - 1;
        int gy = tile_j0 + int(hy) - 1;
        gx = clamp(gx, 0, NRi - 1);
        gy = clamp(gy, 0, NZi - 1);
        tile[hy][hx] = psi_in[gy * NRi + gx];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint i = gid.x;
    uint j = gid.y;
    if (i >= NR || j >= NZ) return;

    uint idx = j * NR + i;

    if (i == 0u || j == 0u || i == NR - 1u || j == NZ - 1u) {
        // Boundary: passthrough. Use tile load to avoid extra device fetch.
        psi_out[idx] = tile[ly + 1u][lx + 1u];
        return;
    }

    float psi_C = tile[ly + 1u][lx + 1u];
    float psi_W = tile[ly + 1u][lx     ];
    float psi_E = tile[ly + 1u][lx + 2u];
    float psi_S = tile[ly     ][lx + 1u];
    float psi_N = tile[ly + 2u][lx + 1u];

    float R         = Rmin + float(i) * dR;
    float inv_dR    = 1.0f / dR;
    float inv_dR2   = inv_dR * inv_dR;
    float inv_dZ2   = 1.0f / (dZ * dZ);
    float h_inv_RdR = 0.5f * inv_dR / R;
    float a_W  = inv_dR2 + h_inv_RdR;
    float a_E  = inv_dR2 - h_inv_RdR;
    float a_NS = inv_dZ2;
    float a_C  = -2.0f * inv_dR2 - 2.0f * inv_dZ2;
    float inv_aC = 1.0f / a_C;

    float ax = psi_axis[0];
    float psi_norm = psi_C / ax;
    float J = 0.0f;
    if (psi_norm > 0.0f && psi_norm < 1.0f) {
        J = R * p_axis * 4.0f * psi_norm * (1.0f - psi_norm);
    }
    float rhs = -mu0 * R * J;

    float delta_psi = a_W * psi_W + a_E * psi_E
                    + a_NS * (psi_N + psi_S)
                    + a_C  * psi_C;
    float r = rhs - delta_psi;
    psi_out[idx] = psi_C + omega * r * inv_aC;
}
```