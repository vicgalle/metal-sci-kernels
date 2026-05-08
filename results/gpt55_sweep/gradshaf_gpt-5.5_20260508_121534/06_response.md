The main change is a vectorized interior max reduction using unaligned `packed_float4` loads, reducing instruction count in the single-threadgroup bottleneck while preserving coalesced row access. The step kernel keeps the proven SIMD-shuffle stencil but trims some per-lane arithmetic by computing R-dependent coefficients only in the source half-row used for broadcast.

```metal
#include <metal_stdlib>
using namespace metal;

static inline float max4f(const float4 v) {
    return max(max(v.x, v.y), max(v.z, v.w));
}

[[max_total_threads_per_threadgroup(256)]]
kernel void gradshaf_axis_reduce(device const float *psi      [[buffer(0)]],
                                 device       float *psi_axis [[buffer(1)]],
                                 constant uint       &NR      [[buffer(2)]],
                                 constant uint       &NZ      [[buffer(3)]],
                                 uint tid                     [[thread_position_in_threadgroup]],
                                 uint tgsize                  [[threads_per_threadgroup]]) {
    threadgroup float partial[8];

    float local_max = -INFINITY;

    if (NR > 2u && NZ > 2u) {
        const uint lx       = tid & 31u;
        const uint ly       = tid >> 5;
        const uint nsg      = (tgsize + 31u) >> 5;
        const uint col_stop = NR - 1u;
        const uint row_stop = NZ - 1u;

        for (uint j = 1u + ly; j < row_stop; j += nsg) {
            const uint base = j * NR;

            // Each SIMD lane owns 4 consecutive columns per vector chunk.
            // packed_float4 permits the naturally unaligned base+1 interior load.
            uint i = 1u + (lx << 2u);

            for (; (i + 3u) < col_stop; i += 128u) {
                const float4 v = float4(*((device const packed_float4 *)(psi + base + i)));
                local_max = max(local_max, max4f(v));
            }

            // Per-lane tail, at most 3 values for the final partial vector.
            for (; i < col_stop; ++i) {
                local_max = max(local_max, psi[base + i]);
            }
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

    const bool valid = (i < NR) && (j < NZ);
    const uint idx = j * NR + i;

    const float psi_C = valid ? psi_in[idx] : 0.0f;

    // With 16-wide threadgroups, each SIMDgroup covers two rows x 16 columns.
    const uint lx   = i & 15u;
    const uint ybit = j & 1u;
    const uint lane = (ybit << 4u) | lx;

    const ushort lane_w = ushort((lx > 0u)    ? (lane - 1u)  : lane);
    const ushort lane_e = ushort((lx < 15u)   ? (lane + 1u)  : lane);
    const ushort lane_s = ushort((ybit > 0u)  ? (lane - 16u) : lane);
    const ushort lane_n = ushort((ybit == 0u) ? (lane + 16u) : lane);

    const float sh_W = simd_shuffle(psi_C, lane_w);
    const float sh_E = simd_shuffle(psi_C, lane_e);
    const float sh_S = simd_shuffle(psi_C, lane_s);
    const float sh_N = simd_shuffle(psi_C, lane_n);

    // One reciprocal per SIMDgroup.
    const float inv_axis_src = (lane == 0u) ? (1.0f / psi_axis[0]) : 0.0f;
    const float inv_axis = simd_shuffle(inv_axis_src, ushort(0));

    float cR = 0.25f;
    float cZ = 0.25f;
    float asym;
    float src;

    if (NR == NZ) {
        const float asym_coeff = 0.125f * dR;
        const float src_coeff  = (mu0 * p_axis) * (dR * dR);

        float asym_src = 0.0f;
        float src_src  = 0.0f;

        // Broadcast column-dependent coefficients from the first row
        // of the 2-row SIMDgroup, avoiding duplicate R math in row 2.
        if (ybit == 0u) {
            const float R = fma(float(i), dR, Rmin);
            asym_src = asym_coeff / R;
            src_src  = src_coeff * (R * R);
        }

        asym = simd_shuffle(asym_src, ushort(lx));
        src  = simd_shuffle(src_src,  ushort(lx));
    } else {
        const float inv_dR  = 1.0f / dR;
        const float inv_dZ  = 1.0f / dZ;
        const float inv_dR2 = inv_dR * inv_dR;
        const float inv_dZ2 = inv_dZ * inv_dZ;
        const float denom   = inv_dR2 + inv_dZ2;
        const float inv_2denom = 0.5f / denom;

        cR = inv_2denom * inv_dR2;
        cZ = inv_2denom * inv_dZ2;

        const float asym_coeff = inv_2denom * (0.5f * inv_dR);
        const float src_coeff  = (2.0f * mu0 * p_axis) / denom;

        float asym_src = 0.0f;
        float src_src  = 0.0f;

        if (ybit == 0u) {
            const float R = fma(float(i), dR, Rmin);
            asym_src = asym_coeff / R;
            src_src  = src_coeff * (R * R);
        }

        asym = simd_shuffle(asym_src, ushort(lx));
        src  = simd_shuffle(src_src,  ushort(lx));
    }

    if (!valid) {
        return;
    }

    if (i == 0u || j == 0u || (i + 1u) == NR || (j + 1u) == NZ) {
        psi_out[idx] = psi_C;
        return;
    }

    const float psi_W = (lx != 0u)    ? sh_W : psi_in[idx - 1u];
    const float psi_E = (lx != 15u)   ? sh_E : psi_in[idx + 1u];
    const float psi_S = (ybit != 0u)  ? sh_S : psi_in[idx - NR];
    const float psi_N = (ybit == 0u)  ? sh_N : psi_in[idx + NR];

    const float psi_norm = psi_C * inv_axis;
    const float raw_shape = psi_norm * (1.0f - psi_norm);
    const float shape = ((psi_norm > 0.0f) && (psi_norm < 1.0f)) ? raw_shape : 0.0f;

    float base = fma(cR, (psi_W + psi_E), cZ * (psi_N + psi_S));
    base = fma(asym, (psi_W - psi_E), base);
    base = fma(src, shape, base);

    if (omega == 1.0f) {
        psi_out[idx] = base;
    } else {
        psi_out[idx] = fma(omega, (base - psi_C), psi_C);
    }
}
```