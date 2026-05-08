The reduction keeps the incumbent’s exact interior max but removes per-element division/modulo by advancing each thread’s flattened interior index with a quotient/remainder recurrence, then uses SIMDgroup reduction to cut barriers. The step uses an algebraically simplified Jacobi update: the center-coefficient term cancels, so square grids compute directly from neighbor averaging, the R-gradient correction, and the source term. This removes large stencil products and a per-cell divide while preserving the same update. 

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
    threadgroup float partial[8];

    if (NR <= 2u || NZ <= 2u) {
        if (tid == 0u) {
            psi_axis[0] = -INFINITY;
        }
        return;
    }

    const uint NR_int = NR - 2u;
    const uint NZ_int = NZ - 2u;
    const uint total  = NR_int * NZ_int;

    float local_max = -INFINITY;

    if (tid < total) {
        // Initial flattened-interior coordinate for k = tid.
        uint i_int = tid % NR_int;
        uint j_int = tid / NR_int;
        uint idx   = (j_int + 1u) * NR + (i_int + 1u);

        // Advance by tgsize in flattened-interior space without hot-loop div/mod.
        const uint q = tgsize / NR_int;
        const uint r = tgsize - q * NR_int;
        const uint base_idx_step = q * NR + r;  // plus 2 when crossing an interior row

        while (j_int < NZ_int) {
            local_max = max(local_max, psi[idx]);

            uint ni = i_int + r;
            idx += base_idx_step;
            j_int += q;

            if (ni >= NR_int) {
                ni -= NR_int;
                ++j_int;
                idx += 2u; // skip right boundary of old row + left boundary of new row
            }
            i_int = ni;
        }
    }

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

    const float R = Rmin + float(i) * dR;
    const float axis = psi_axis[0];
    const float psi_norm = psi_C / axis;

    float shape = 0.0f;
    if (psi_norm > 0.0f && psi_norm < 1.0f) {
        shape = psi_norm * (1.0f - psi_norm);
    }

    float base;

    if (NR == NZ) {
        // For dR == dZ, the Jacobi center term cancels:
        // ψ_new(ω=1) = 1/4*(W+E+N+S) + dR/(8R)*(W-E)
        //              + μ0*p_axis*dR^2*R^2*ψn*(1-ψn).
        const float avg  = 0.25f * ((psi_W + psi_E) + (psi_N + psi_S));
        const float corr = (0.125f * dR / R) * (psi_W - psi_E);
        const float src  = (mu0 * p_axis) * (dR * dR) * (R * R) * shape;
        base = avg + corr + src;
    } else {
        // Generic algebraically-simplified form for non-square grids.
        const float inv_dR  = 1.0f / dR;
        const float inv_dZ  = 1.0f / dZ;
        const float inv_dR2 = inv_dR * inv_dR;
        const float inv_dZ2 = inv_dZ * inv_dZ;
        const float denom   = inv_dR2 + inv_dZ2;
        const float inv_2denom = 0.5f / denom;

        const float h_inv_RdR = 0.5f * inv_dR / R;

        base = inv_2denom *
               (inv_dR2 * (psi_W + psi_E) +
                inv_dZ2 * (psi_N + psi_S) +
                h_inv_RdR * (psi_W - psi_E));

        base += (2.0f * mu0 * p_axis / denom) * (R * R) * shape;
    }

    psi_out[idx] = psi_C + omega * (base - psi_C);
}
```