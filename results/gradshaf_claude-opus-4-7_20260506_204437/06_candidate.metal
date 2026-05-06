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

    // Iterate row-by-row over interior; each thread strides across columns.
    // This gives perfectly coalesced loads within each row.
    for (uint jj = 0u; jj < NZ_int; ++jj) {
        uint row_base = (jj + 1u) * NR + 1u;
        uint k = tid;
        // Unroll by 4 along the row.
        for (; k + 4u * tgsize <= NR_int; k += 4u * tgsize) {
            float v0 = psi[row_base + k];
            float v1 = psi[row_base + k + tgsize];
            float v2 = psi[row_base + k + 2u * tgsize];
            float v3 = psi[row_base + k + 3u * tgsize];
            local_max = max(local_max, max(max(v0, v1), max(v2, v3)));
        }
        for (; k < NR_int; k += tgsize) {
            local_max = max(local_max, psi[row_base + k]);
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
                          uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint j = gid.y;
    if (i >= NR || j >= NZ) return;

    uint idx = j * NR + i;

    bool is_boundary = (i == 0u) | (j == 0u) | (i == NR - 1u) | (j == NZ - 1u);

    if (is_boundary) {
        psi_out[idx] = psi_in[idx];
        return;
    }

    // 5-point stencil reads — rely on L1/L2 cache.
    float psi_C = psi_in[idx];
    float psi_W = psi_in[idx - 1u];
    float psi_E = psi_in[idx + 1u];
    float psi_S = psi_in[idx - NR];
    float psi_N = psi_in[idx + NR];

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

    // Read axis once; use reciprocal multiply instead of divide.
    float ax = psi_axis[0];
    float inv_ax = 1.0f / ax;
    float psi_norm_raw = psi_C * inv_ax;
    // Branchless J: clamp psi_norm to [0,1]; 4*x*(1-x) is 0 at endpoints,
    // so values outside (0,1) yield J = 0 as required.
    float psi_norm = clamp(psi_norm_raw, 0.0f, 1.0f);
    float J = R * p_axis * 4.0f * psi_norm * (1.0f - psi_norm);
    float rhs = -mu0 * R * J;

    float delta_psi = fma(a_W, psi_W,
                       fma(a_E, psi_E,
                        fma(a_NS, psi_N + psi_S,
                          a_C * psi_C)));
    float r = rhs - delta_psi;
    psi_out[idx] = fma(omega * r, inv_aC, psi_C);
}