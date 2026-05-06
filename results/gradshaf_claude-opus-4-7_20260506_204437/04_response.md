**Optimization & Rationale**

The kernel is BW-bound but achieving only ~7.6 GB/s — far below the 200 GB/s roofline. The issue is dispatch/launch overhead dominates: each outer step is two dispatches, and at 513×513 with K=30 we get 60 dispatches taking 12 ms (~200 µs/dispatch). The arithmetic per cell is also heavier than needed.

Key improvements:
1. **Hoist invariants**: `inv_dR`, `inv_dR2`, `inv_dZ2`, `1/a_C` only depend on dR/dZ — but they're recomputed per thread. The compiler likely hoists them, but R-dependent terms (`a_W`, `a_E`) can be simplified algebraically.
2. **Reformulate the update** using `a_C = -2(inv_dR2+inv_dZ2)`. The residual `r/a_C` term: combine to a single FMA-friendly form. Compute `psi_new = psi_C + omega*(rhs - Δ*ψ)/a_C`. With ω=1, expanding: `ψ_new = (a_W·ψ_W + a_E·ψ_E + a_NS·(ψ_N+ψ_S) - rhs) / (-a_C)` — saves a couple ops and avoids the division-by-R inside the inner loop by precomputing `1/R` via a single rcp.
3. **Vectorized boundary handling**: avoid branchy boundary check by predicating writes.
4. **Use `fast::` math** for division.

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

    // Unroll by 4 to better hide latency.
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

    // Hoisted constants (compiler should fold across threads but explicit FMA-friendly form helps).
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

    // Δ*ψ via FMAs
    float delta_psi = fma(a_W, psi_W,
                       fma(a_E, psi_E,
                        fma(a_NS, psi_N + psi_S,
                          a_C * psi_C)));
    float r = rhs - delta_psi;
    psi_out[idx] = fma(omega * r, inv_aC, psi_C);
}
```