**Optimization:** Add threadgroup-memory tiling for the stencil step. With a (16,16) TG, load a 18×18 tile (16×16 interior + halo) cooperatively into threadgroup memory, then each thread computes its stencil from the tile. This converts 5 device loads per cell into 1 cooperative load (~1.4 reads/cell amortized), which should help on the 257² and 513² cases where the kernel is memory-bound. Boundary cells still copy through. The reduce kernel keeps its strided unrolled form. Halo loads are guarded against out-of-bounds (the dispatch is rounded up to multiples of 16, and we only need the halo within [0, NR-1] / [0, NZ-1] since interior cells are i,j ∈ [1, N-1)). For the smallest grid (65²) global-memory caching may already be sufficient, but the larger grids should benefit most.

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

    uint k = tid;
    uint stride = tgsize;
    uint stride4 = stride * 4u;
    for (; k + stride4 <= total; k += stride4) {
        uint k0 = k;
        uint k1 = k + stride;
        uint k2 = k + 2u * stride;
        uint k3 = k + 3u * stride;
        uint j0 = k0 / NR_int, i0 = k0 - j0 * NR_int;
        uint j1 = k1 / NR_int, i1 = k1 - j1 * NR_int;
        uint j2 = k2 / NR_int, i2 = k2 - j2 * NR_int;
        uint j3 = k3 / NR_int, i3 = k3 - j3 * NR_int;
        float v0 = psi[(j0 + 1u) * NR + (i0 + 1u)];
        float v1 = psi[(j1 + 1u) * NR + (i1 + 1u)];
        float v2 = psi[(j2 + 1u) * NR + (i2 + 1u)];
        float v3 = psi[(j3 + 1u) * NR + (i3 + 1u)];
        local_max = max(local_max, max(max(v0, v1), max(v2, v3)));
    }
    for (; k < total; k += stride) {
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
                          uint2 ltid [[thread_position_in_threadgroup]],
                          uint2 tgid [[threadgroup_position_in_grid]]) {
    threadgroup float tile[HALO_H][HALO_W];

    uint i = gid.x;
    uint j = gid.y;

    int tile_i0 = int(tgid.x) * TILE_W - 1; // top-left of halo region in global coords
    int tile_j0 = int(tgid.y) * TILE_H - 1;

    uint lx = ltid.x;
    uint ly = ltid.y;
    uint local_lin = ly * TILE_W + lx; // 0..255

    // Cooperative load of HALO_H * HALO_W = 18*18 = 324 floats with 256 threads.
    // Each thread loads up to 2 elements.
    int NRi = int(NR);
    int NZi = int(NZ);

    for (uint n = local_lin; n < HALO_H * HALO_W; n += TILE_W * TILE_H) {
        uint ty = n / HALO_W;
        uint tx = n - ty * HALO_W;
        int gx = tile_i0 + int(tx);
        int gy = tile_j0 + int(ty);
        float v = 0.0f;
        if (gx >= 0 && gx < NRi && gy >= 0 && gy < NZi) {
            v = psi_in[gy * NRi + gx];
        }
        tile[ty][tx] = v;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (i >= NR || j >= NZ) return;

    uint idx = j * NR + i;

    bool is_boundary = (i == 0u) || (j == 0u) || (i == NR - 1u) || (j == NZ - 1u);
    if (is_boundary) {
        psi_out[idx] = psi_in[idx];
        return;
    }

    // Local tile coords for this thread's center: (lx+1, ly+1)
    uint tlx = lx + 1u;
    uint tly = ly + 1u;
    float psi_C = tile[tly][tlx];
    float psi_W = tile[tly][tlx - 1u];
    float psi_E = tile[tly][tlx + 1u];
    float psi_S = tile[tly - 1u][tlx];
    float psi_N = tile[tly + 1u][tlx];

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