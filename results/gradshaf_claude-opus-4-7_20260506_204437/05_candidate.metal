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

// Tile is sized to the dispatched threadgroup shape (16 x 16) plus a 1-cell halo
// on each side: 18 x 18.
constant constexpr uint TW = 16;
constant constexpr uint TH = 16;
constant constexpr uint SW = TW + 2;  // 18
constant constexpr uint SH = TH + 2;  // 18

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
    threadgroup float tile[SH * SW];

    uint i = gid.x;
    uint j = gid.y;

    // Tile origin in global coords (matches dispatched TG = 16x16).
    int tile_i0 = int(tgid.x) * int(TW);
    int tile_j0 = int(tgid.y) * int(TH);

    uint lx = lid.x;
    uint ly = lid.y;
    uint flat = ly * TW + lx;          // 0..255
    uint tg_threads = TW * TH;         // 256

    // Cooperative load of 18x18 = 324 floats with 256 threads -> 2 passes.
    int NRi = int(NR);
    int NZi = int(NZ);
    for (uint p = flat; p < SW * SH; p += tg_threads) {
        uint ty = p / SW;
        uint tx = p - ty * SW;
        int gx = tile_i0 + int(tx) - 1;
        int gy = tile_j0 + int(ty) - 1;
        // Clamp to [0, N-1]; boundary threads will not use stale halo since they
        // bypass the stencil. Interior threads always have valid neighbors.
        gx = clamp(gx, 0, NRi - 1);
        gy = clamp(gy, 0, NZi - 1);
        tile[p] = psi_in[gy * NRi + gx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (i >= NR || j >= NZ) return;

    uint idx = j * NR + i;

    bool is_boundary = (i == 0u) || (j == 0u) || (i == NR - 1u) || (j == NZ - 1u);
    if (is_boundary) {
        // Direct copy from tile center (which equals psi_in[idx] post-clamp,
        // since clamp leaves valid cells unchanged).
        psi_out[idx] = tile[(ly + 1u) * SW + (lx + 1u)];
        return;
    }

    // Read 5-point stencil from threadgroup memory.
    uint c = (ly + 1u) * SW + (lx + 1u);
    float psi_C = tile[c];
    float psi_W = tile[c - 1u];
    float psi_E = tile[c + 1u];
    float psi_S = tile[c - SW];
    float psi_N = tile[c + SW];

    float inv_dR    = 1.0f / dR;
    float inv_dR2   = inv_dR * inv_dR;
    float inv_dZ2   = 1.0f / (dZ * dZ);
    float R         = fma(float(i), dR, Rmin);
    float inv_R     = 1.0f / R;
    float h_inv_RdR = 0.5f * inv_dR * inv_R;
    float a_W  = inv_dR2 + h_inv_RdR;
    float a_E  = inv_dR2 - h_inv_RdR;
    float a_NS = inv_dZ2;
    float a_C  = -2.0f * (inv_dR2 + inv_dZ2);
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