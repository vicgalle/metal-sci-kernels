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
            uint i = 1u + (lx << 2u);

            for (; (i + 387u) < col_stop; i += 512u) {
                const float4 v0 = float4(*((device const packed_float4 *)(psi + base + i)));
                const float4 v1 = float4(*((device const packed_float4 *)(psi + base + i + 128u)));
                const float4 v2 = float4(*((device const packed_float4 *)(psi + base + i + 256u)));
                const float4 v3 = float4(*((device const packed_float4 *)(psi + base + i + 384u)));
                local_max = max(local_max, max(max4f(v0), max4f(v1)));
                local_max = max(local_max, max(max4f(v2), max4f(v3)));
            }

            for (; (i + 131u) < col_stop; i += 256u) {
                const float4 v0 = float4(*((device const packed_float4 *)(psi + base + i)));
                const float4 v1 = float4(*((device const packed_float4 *)(psi + base + i + 128u)));
                local_max = max(local_max, max(max4f(v0), max4f(v1)));
            }

            for (; (i + 3u) < col_stop; i += 128u) {
                const float4 v = float4(*((device const packed_float4 *)(psi + base + i)));
                local_max = max(local_max, max4f(v));
            }

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
    threadgroup float vtile[18u * 16u];
    threadgroup float tg_asym[16u];
    threadgroup float tg_src[16u];
    threadgroup float tg_inv_axis;

    const uint i  = gid.x;
    const uint j  = gid.y;
    const uint lx = i & 15u;
    const uint ly = j & 15u;

    const bool valid = (i < NR) && (j < NZ);
    const uint idx = j * NR + i;

    const float psi_C = valid ? psi_in[idx] : 0.0f;

    vtile[(ly + 1u) * 16u + lx] = psi_C;

    if (ly == 0u) {
        vtile[lx] = (i < NR && j > 0u && j < NZ) ? psi_in[idx - NR] : 0.0f;

        if (i < NR) {
            const float R = fma(float(i), dR, Rmin);

            if (NR == NZ) {
                const float asym_coeff = 0.125f * dR;
                const float src_coeff  = (mu0 * p_axis) * (dR * dR);
                tg_asym[lx] = asym_coeff / R;
                tg_src[lx]  = src_coeff * (R * R);
            } else {
                const float inv_dR  = 1.0f / dR;
                const float inv_dZ  = 1.0f / dZ;
                const float inv_dR2 = inv_dR * inv_dR;
                const float inv_dZ2 = inv_dZ * inv_dZ;
                const float denom   = inv_dR2 + inv_dZ2;
                const float inv_2denom = 0.5f / denom;

                const float asym_coeff = inv_2denom * (0.5f * inv_dR);
                const float src_coeff  = (2.0f * mu0 * p_axis) / denom;
                tg_asym[lx] = asym_coeff / R;
                tg_src[lx]  = src_coeff * (R * R);
            }
        } else {
            tg_asym[lx] = 0.0f;
            tg_src[lx]  = 0.0f;
        }

        if (lx == 0u) {
            tg_inv_axis = 1.0f / psi_axis[0];
        }
    }

    if (ly == 15u) {
        vtile[17u * 16u + lx] = (i < NR && (j + 1u) < NZ) ? psi_in[idx + NR] : 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint ybit = j & 1u;
    const uint lane = (ybit << 4u) | lx;

    const ushort lane_w = ushort((lx > 0u)  ? (lane - 1u) : lane);
    const ushort lane_e = ushort((lx < 15u) ? (lane + 1u) : lane);

    const float sh_W = simd_shuffle(psi_C, lane_w);
    const float sh_E = simd_shuffle(psi_C, lane_e);

    if (!valid) {
        return;
    }

    if (i == 0u || j == 0u || (i + 1u) == NR || (j + 1u) == NZ) {
        psi_out[idx] = psi_C;
        return;
    }

    const float psi_W = (lx != 0u)  ? sh_W : psi_in[idx - 1u];
    const float psi_E = (lx != 15u) ? sh_E : psi_in[idx + 1u];
    const float psi_S = vtile[ly * 16u + lx];
    const float psi_N = vtile[(ly + 2u) * 16u + lx];

    float cR = 0.25f;
    float cZ = 0.25f;

    if (NR != NZ) {
        const float inv_dR  = 1.0f / dR;
        const float inv_dZ  = 1.0f / dZ;
        const float inv_dR2 = inv_dR * inv_dR;
        const float inv_dZ2 = inv_dZ * inv_dZ;
        const float denom   = inv_dR2 + inv_dZ2;
        const float inv_2denom = 0.5f / denom;

        cR = inv_2denom * inv_dR2;
        cZ = inv_2denom * inv_dZ2;
    }

    const float psi_norm = psi_C * tg_inv_axis;
    const float raw_shape = psi_norm * (1.0f - psi_norm);
    const float shape = ((psi_norm > 0.0f) && (psi_norm < 1.0f)) ? raw_shape : 0.0f;

    float base = fma(cR, (psi_W + psi_E), cZ * (psi_N + psi_S));
    base = fma(tg_asym[lx], (psi_W - psi_E), base);
    base = fma(tg_src[lx], shape, base);

    if (omega == 1.0f) {
        psi_out[idx] = base;
    } else {
        psi_out[idx] = fma(omega, (base - psi_C), psi_C);
    }
}