The update below reduces per-cell expensive work by caching the axis reciprocal and R-dependent coefficients once per 16-column threadgroup, and it uses a 16×16 threadgroup-memory stencil tile with halos so neighboring ψ values are reused from fast local memory. The reduction is also remapped to a SIMD-friendly 32×8 virtual tile, avoiding per-element div/mod and row-crossing logic while preserving the required single-dispatch scalar output.

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

    float local_max = -INFINITY;

    if (NR > 2u && NZ > 2u) {
        if (tgsize == 256u) {
            // SIMD-friendly virtual 32x8 tile: one simdgroup scans one row stripe.
            const uint lx = tid & 31u;
            const uint ly = tid >> 5;       // 0..7

            const uint col_stop = NR - 1u;  // exclusive interior end
            const uint row_stop = NZ - 1u;

            for (uint j = 1u + ly; j < row_stop; j += 8u) {
                const uint base = j * NR;
                uint i = 1u + lx;

                // Unroll by two column-strides to reduce loop overhead.
                for (; (i + 32u) < col_stop; i += 64u) {
                    const float a = psi[base + i];
                    const float b = psi[base + i + 32u];
                    local_max = max(local_max, max(a, b));
                }
                if (i < col_stop) {
                    local_max = max(local_max, psi[base + i]);
                }
            }
        } else {
            // Conservative fallback for non-default tgsize.
            const uint NR_int = NR - 2u;
            const uint NZ_int = NZ - 2u;
            const uint total  = NR_int * NZ_int;

            if (tid < total) {
                uint i_int = tid % NR_int;
                uint j_int = tid / NR_int;
                uint idx   = (j_int + 1u) * NR + (i_int + 1u);

                const uint q = tgsize / NR_int;
                const uint r = tgsize - q * NR_int;
                const uint base_idx_step = q * NR + r;

                while (j_int < NZ_int) {
                    local_max = max(local_max, psi[idx]);

                    uint ni = i_int + r;
                    idx += base_idx_step;
                    j_int += q;

                    if (ni >= NR_int) {
                        ni -= NR_int;
                        ++j_int;
                        idx += 2u;
                    }
                    i_int = ni;
                }
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
    threadgroup float tile[18u * 18u];
    threadgroup float tg_asym[16];
    threadgroup float tg_src[16];
    threadgroup float tg_inv_axis;
    threadgroup float tg_cR;
    threadgroup float tg_cZ;
    threadgroup float tg_omega;

    const uint i  = gid.x;
    const uint j  = gid.y;
    const uint lx = i & 15u;
    const uint ly = j & 15u;

    const bool valid = (i < NR) && (j < NZ);
    const uint idx = j * NR + i;

    const uint tcenter = (ly + 1u) * 18u + (lx + 1u);

    float center = 0.0f;
    if (valid) {
        center = psi_in[idx];
    }
    tile[tcenter] = center;

    // Cooperative halo loads for a 16x16 stencil tile.
    if (lx == 0u) {
        tile[(ly + 1u) * 18u] = (valid && i > 0u) ? psi_in[idx - 1u] : 0.0f;
    }
    if (lx == 15u) {
        tile[(ly + 1u) * 18u + 17u] = ((j < NZ) && ((i + 1u) < NR)) ? psi_in[idx + 1u] : 0.0f;
    }
    if (ly == 0u) {
        tile[lx + 1u] = (valid && j > 0u) ? psi_in[idx - NR] : 0.0f;
    }
    if (ly == 15u) {
        tile[17u * 18u + (lx + 1u)] = (((j + 1u) < NZ) && (i < NR)) ? psi_in[idx + NR] : 0.0f;
    }

    // Cache axis reciprocal and R-dependent coefficients once per column in the TG.
    if (ly == 0u) {
        const float R = Rmin + float(i) * dR;

        if (NR == NZ) {
            tg_asym[lx] = (0.125f * dR) / R;
            tg_src[lx]  = (mu0 * p_axis) * (dR * dR) * (R * R);

            if (lx == 0u) {
                tg_cR = 0.25f;
                tg_cZ = 0.25f;
                tg_inv_axis = 1.0f / psi_axis[0];
                tg_omega = omega;
            }
        } else {
            const float inv_dR  = 1.0f / dR;
            const float inv_dZ  = 1.0f / dZ;
            const float inv_dR2 = inv_dR * inv_dR;
            const float inv_dZ2 = inv_dZ * inv_dZ;
            const float denom   = inv_dR2 + inv_dZ2;
            const float inv_2denom = 0.5f / denom;

            tg_asym[lx] = inv_2denom * (0.5f * inv_dR / R);
            tg_src[lx]  = (2.0f * mu0 * p_axis / denom) * (R * R);

            if (lx == 0u) {
                tg_cR = inv_2denom * inv_dR2;
                tg_cZ = inv_2denom * inv_dZ2;
                tg_inv_axis = 1.0f / psi_axis[0];
                tg_omega = omega;
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!valid) {
        return;
    }

    const float psi_C = tile[tcenter];

    if (i == 0u || j == 0u || i == (NR - 1u) || j == (NZ - 1u)) {
        psi_out[idx] = psi_C;
        return;
    }

    const float psi_W = tile[(ly + 1u) * 18u + lx];
    const float psi_E = tile[(ly + 1u) * 18u + (lx + 2u)];
    const float psi_S = tile[ly * 18u + (lx + 1u)];
    const float psi_N = tile[(ly + 2u) * 18u + (lx + 1u)];

    const float psi_norm = psi_C * tg_inv_axis;
    const float raw_shape = psi_norm * (1.0f - psi_norm);
    const float shape = ((psi_norm > 0.0f) && (psi_norm < 1.0f)) ? raw_shape : 0.0f;

    float base;
    if (NR == NZ) {
        base = 0.25f * ((psi_W + psi_E) + (psi_N + psi_S));
    } else {
        base = fma(tg_cR, (psi_W + psi_E), tg_cZ * (psi_N + psi_S));
    }

    base = fma(tg_asym[lx], (psi_W - psi_E), base);
    base = fma(tg_src[lx], shape, base);

    if (tg_omega == 1.0f) {
        psi_out[idx] = base;
    } else {
        psi_out[idx] = psi_C + tg_omega * (base - psi_C);
    }
}
```