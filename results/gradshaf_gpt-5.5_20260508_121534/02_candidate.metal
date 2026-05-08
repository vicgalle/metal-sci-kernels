#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(256)]]
kernel void gradshaf_axis_reduce(device const float *psi      [[buffer(0)]],
                                 device       float *psi_axis [[buffer(1)]],
                                 constant uint       &NR      [[buffer(2)]],
                                 constant uint       &NZ      [[buffer(3)]],
                                 uint tid                     [[thread_position_in_threadgroup]],
                                 uint tgsize                  [[threads_per_threadgroup]]) {
    threadgroup float partial[8];

    if (NR <= 2u || NZ <= 2u) {
        if (tid == 0u) {
            psi_axis[0] = -INFINITY;
        }
        return;
    }

    const uint NRm1 = NR - 1u;
    const uint NZm1 = NZ - 1u;

    float local_max = -INFINITY;

    // Row-wise sweep avoids expensive k / NR_int and k % NR_int in the hot loop.
    for (uint j = 1u; j < NZm1; ++j) {
        const uint base = j * NR;

        for (uint i = tid + 1u; i < NRm1; i += tgsize) {
            local_max = max(local_max, psi[base + i]);
        }
    }

    // Reduce within each SIMDgroup, then reduce SIMDgroup maxima with SIMDgroup 0.
    const float sg_max = simd_max(local_max);
    if ((tid & 31u) == 0u) {
        partial[tid >> 5] = sg_max;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32u) {
        const uint nsg = (tgsize + 31u) >> 5;
        float v = (tid < nsg) ? partial[tid] : -INFINITY;
        v = simd_max(v);
        if (tid == 0u) {
            psi_axis[0] = v;
        }
    }
}

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
    const uint i = gid.x;
    const uint j = gid.y;

    if (i >= NR || j >= NZ) {
        return;
    }

    const uint idx = j * NR + i;
    const float psi_C = psi_in[idx];

    if (i == 0u || j == 0u || i == NR - 1u || j == NZ - 1u) {
        psi_out[idx] = psi_C;
        return;
    }

    const float psi_W = psi_in[idx - 1u];
    const float psi_E = psi_in[idx + 1u];
    const float psi_S = psi_in[idx - NR];
    const float psi_N = psi_in[idx + NR];

    // Fixed unit extents in R and Z: dR = 1/(NR-1), dZ = 1/(NZ-1).
    // This removes two reciprocal/divide chains per cell.
    const float inv_dR  = float(NR - 1u);
    const float inv_dZ  = float(NZ - 1u);
    const float inv_dR2 = inv_dR * inv_dR;
    const float inv_dZ2 = inv_dZ * inv_dZ;

    const float R = Rmin + float(i) * dR;
    const float half_inv_RdR = (0.5f * inv_dR) / R;

    const float a_C = -2.0f * (inv_dR2 + inv_dZ2);
    const float inv_a_C = 1.0f / a_C;

    // Algebraically equivalent 5-point operator:
    // inv_dR2*(W+E-2C) + inv_dZ2*(N+S-2C) + (1/(2RdR))*(W-E)
    const float delta_psi =
        inv_dR2 * (psi_W + psi_E - 2.0f * psi_C) +
        inv_dZ2 * (psi_N + psi_S - 2.0f * psi_C) +
        half_inv_RdR * (psi_W - psi_E);

    const float psi_norm = psi_C / psi_axis[0];

    float rhs = 0.0f;
    if (psi_norm > 0.0f && psi_norm < 1.0f) {
        const float shape = psi_norm * (1.0f - psi_norm);
        rhs = -(4.0f * mu0 * p_axis) * (R * R) * shape;
    }

    psi_out[idx] = psi_C + omega * (rhs - delta_psi) * inv_a_C;
}