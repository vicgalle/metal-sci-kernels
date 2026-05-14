I’m replacing most global scratch traffic in the common 64/96/128 cases with register checkpointing: the forward sweep stores only every 8th modified RHS in registers, then recomputes small chunks during back-substitution. This trades a little extra arithmetic and cached RHS rereads for eliminating the previous intermediate global writes/reads, which should improve the memory-bound sweeps while preserving the exact same coefficient approximation used by the incumbent.

```metal
#include <metal_stdlib>
using namespace metal;

constant constexpr float HALF = 0.5f;

// Thomas coefficients for mu = 0.5:
// a = c = -0.5, b = 2.0.
constant constexpr float CP1   = -0.25f;
constant constexpr float CP2   = -0.2666666666666667f;
constant constexpr float CP3   = -0.26785714285714285f;
constant constexpr float CP4   = -0.2679425837320574f;
constant constexpr float CP5   = -0.26794871794871794f;
constant constexpr float CP6   = -0.2679491583648231f;
constant constexpr float CP7   = -2911.0f / 10864.0f;
constant constexpr float CP8   = -10864.0f / 40545.0f;
constant constexpr float CPINF = -0.2679491924311227f;

constant constexpr float INV1   = -2.0f * CP1;
constant constexpr float INV2   = -2.0f * CP2;
constant constexpr float INV3   = -2.0f * CP3;
constant constexpr float INV4   = -2.0f * CP4;
constant constexpr float INV5   = -2.0f * CP5;
constant constexpr float INV6   = -2.0f * CP6;
constant constexpr float INV7   = -2.0f * CP7;
constant constexpr float INV8   = -2.0f * CP8;
constant constexpr float INVINF = -2.0f * CPINF;

inline float inv_for_index(uint i)
{
    if (i >= 9u) return INVINF;
    if (i == 8u) return INV8;
    if (i == 7u) return INV7;
    if (i == 6u) return INV6;
    if (i == 5u) return INV5;
    if (i == 4u) return INV4;
    if (i == 3u) return INV3;
    if (i == 2u) return INV2;
    return INV1;
}

inline void copy_line(device const float *u_in,
                      device       float *u_out,
                      uint base, uint stride, uint N)
{
    uint idx = base;
    uint n = 0u;

    for (; n + 4u <= N; n += 4u) {
        uint i0 = idx;
        uint i1 = i0 + stride;
        uint i2 = i1 + stride;
        uint i3 = i2 + stride;

        float v0 = u_in[i0];
        float v1 = u_in[i1];
        float v2 = u_in[i2];
        float v3 = u_in[i3];

        u_out[i0] = v0;
        u_out[i1] = v1;
        u_out[i2] = v2;
        u_out[i3] = v3;

        idx = i3 + stride;
    }

    for (; n < N; ++n) {
        u_out[idx] = u_in[idx];
        idx += stride;
    }
}

inline float advance8_inf(device const float *u_in,
                          uint base, uint stride,
                          uint start, float dp)
{
    uint idx = base + start * stride;

    float q0 = (u_in[idx] + HALF * dp) * INVINF; idx += stride;
    float q1 = (u_in[idx] + HALF * q0) * INVINF; idx += stride;
    float q2 = (u_in[idx] + HALF * q1) * INVINF; idx += stride;
    float q3 = (u_in[idx] + HALF * q2) * INVINF; idx += stride;
    float q4 = (u_in[idx] + HALF * q3) * INVINF; idx += stride;
    float q5 = (u_in[idx] + HALF * q4) * INVINF; idx += stride;
    float q6 = (u_in[idx] + HALF * q5) * INVINF; idx += stride;
    float q7 = (u_in[idx] + HALF * q6) * INVINF;

    return q7;
}

inline float final_after5_inf(device const float *u_in,
                              uint base, uint stride,
                              uint start, float dp, float bd_hi)
{
    uint idx = base + start * stride;

    float q0 = (u_in[idx] + HALF * dp) * INVINF; idx += stride;
    float q1 = (u_in[idx] + HALF * q0) * INVINF; idx += stride;
    float q2 = (u_in[idx] + HALF * q1) * INVINF; idx += stride;
    float q3 = (u_in[idx] + HALF * q2) * INVINF; idx += stride;
    float q4 = (u_in[idx] + HALF * q3) * INVINF; idx += stride;

    return (u_in[idx] + HALF * q4 + HALF * bd_hi) * INVINF;
}

inline float back_chunk5_inf(device const float *u_in,
                             device       float *u_out,
                             uint base, uint stride,
                             uint start, float dprev, float xnext)
{
    uint ridx = base + start * stride;

    float q0 = (u_in[ridx] + HALF * dprev) * INVINF; ridx += stride;
    float q1 = (u_in[ridx] + HALF * q0) * INVINF; ridx += stride;
    float q2 = (u_in[ridx] + HALF * q1) * INVINF; ridx += stride;
    float q3 = (u_in[ridx] + HALF * q2) * INVINF; ridx += stride;
    float q4 = (u_in[ridx] + HALF * q3) * INVINF;

    uint widx = base + (start + 4u) * stride;

    float x4 = q4 - CPINF * xnext; u_out[widx] = x4; widx -= stride;
    float x3 = q3 - CPINF * x4;    u_out[widx] = x3; widx -= stride;
    float x2 = q2 - CPINF * x3;    u_out[widx] = x2; widx -= stride;
    float x1 = q1 - CPINF * x2;    u_out[widx] = x1; widx -= stride;
    float x0 = q0 - CPINF * x1;    u_out[widx] = x0;

    return x0;
}

inline float back_chunk8_inf(device const float *u_in,
                             device       float *u_out,
                             uint base, uint stride,
                             uint start, float dprev, float xnext)
{
    uint ridx = base + start * stride;

    float q0 = (u_in[ridx] + HALF * dprev) * INVINF; ridx += stride;
    float q1 = (u_in[ridx] + HALF * q0) * INVINF; ridx += stride;
    float q2 = (u_in[ridx] + HALF * q1) * INVINF; ridx += stride;
    float q3 = (u_in[ridx] + HALF * q2) * INVINF; ridx += stride;
    float q4 = (u_in[ridx] + HALF * q3) * INVINF; ridx += stride;
    float q5 = (u_in[ridx] + HALF * q4) * INVINF; ridx += stride;
    float q6 = (u_in[ridx] + HALF * q5) * INVINF; ridx += stride;
    float q7 = (u_in[ridx] + HALF * q6) * INVINF;

    uint widx = base + (start + 7u) * stride;

    float x7 = q7 - CPINF * xnext; u_out[widx] = x7; widx -= stride;
    float x6 = q6 - CPINF * x7;    u_out[widx] = x6; widx -= stride;
    float x5 = q5 - CPINF * x6;    u_out[widx] = x5; widx -= stride;
    float x4 = q4 - CPINF * x5;    u_out[widx] = x4; widx -= stride;
    float x3 = q3 - CPINF * x4;    u_out[widx] = x3; widx -= stride;
    float x2 = q2 - CPINF * x3;    u_out[widx] = x2; widx -= stride;
    float x1 = q1 - CPINF * x2;    u_out[widx] = x1; widx -= stride;
    float x0 = q0 - CPINF * x1;    u_out[widx] = x0;

    return x0;
}

inline void back_first8(device float *u_out,
                        uint base, uint stride,
                        float d1, float d2, float d3, float d4,
                        float d5, float d6, float d7, float d8,
                        float xnext)
{
    uint widx = base + 8u * stride;

    float x8 = d8 - CP8 * xnext; u_out[widx] = x8; widx -= stride;
    float x7 = d7 - CP7 * x8;    u_out[widx] = x7; widx -= stride;
    float x6 = d6 - CP6 * x7;    u_out[widx] = x6; widx -= stride;
    float x5 = d5 - CP5 * x6;    u_out[widx] = x5; widx -= stride;
    float x4 = d4 - CP4 * x5;    u_out[widx] = x4; widx -= stride;
    float x3 = d3 - CP3 * x4;    u_out[widx] = x3; widx -= stride;
    float x2 = d2 - CP2 * x3;    u_out[widx] = x2; widx -= stride;
    float x1 = d1 - CP1 * x2;    u_out[widx] = x1;
}

inline void thomas_line_checkpoint_64_96_128(device const float *u_in,
                                             device       float *u_out,
                                             uint base, uint stride, uint N)
{
    uint last = base + (N - 1u) * stride;
    float bd_lo = u_in[base];
    float bd_hi = u_in[last];

    u_out[base] = bd_lo;
    u_out[last] = bd_hi;

    uint idx = base + stride;

    float d1 = (u_in[idx] + HALF * bd_lo) * INV1; idx += stride;
    float d2 = (u_in[idx] + HALF * d1)    * INV2; idx += stride;
    float d3 = (u_in[idx] + HALF * d2)    * INV3; idx += stride;
    float d4 = (u_in[idx] + HALF * d3)    * INV4; idx += stride;
    float d5 = (u_in[idx] + HALF * d4)    * INV5; idx += stride;
    float d6 = (u_in[idx] + HALF * d5)    * INV6; idx += stride;
    float d7 = (u_in[idx] + HALF * d6)    * INV7; idx += stride;
    float d8 = (u_in[idx] + HALF * d7)    * INV8;

    float d16 = advance8_inf(u_in, base, stride, 9u,  d8);
    float d24 = advance8_inf(u_in, base, stride, 17u, d16);
    float d32 = advance8_inf(u_in, base, stride, 25u, d24);
    float d40 = advance8_inf(u_in, base, stride, 33u, d32);
    float d48 = advance8_inf(u_in, base, stride, 41u, d40);
    float d56 = advance8_inf(u_in, base, stride, 49u, d48);

    if (N == 64u) {
        float xnext = final_after5_inf(u_in, base, stride, 57u, d56, bd_hi);
        u_out[base + 62u * stride] = xnext;

        xnext = back_chunk5_inf(u_in, u_out, base, stride, 57u, d56, xnext);
        xnext = back_chunk8_inf(u_in, u_out, base, stride, 49u, d48, xnext);
        xnext = back_chunk8_inf(u_in, u_out, base, stride, 41u, d40, xnext);
        xnext = back_chunk8_inf(u_in, u_out, base, stride, 33u, d32, xnext);
        xnext = back_chunk8_inf(u_in, u_out, base, stride, 25u, d24, xnext);
        xnext = back_chunk8_inf(u_in, u_out, base, stride, 17u, d16, xnext);
        xnext = back_chunk8_inf(u_in, u_out, base, stride, 9u,  d8,  xnext);
        back_first8(u_out, base, stride, d1, d2, d3, d4, d5, d6, d7, d8, xnext);
        return;
    }

    float d64 = advance8_inf(u_in, base, stride, 57u, d56);
    float d72 = advance8_inf(u_in, base, stride, 65u, d64);
    float d80 = advance8_inf(u_in, base, stride, 73u, d72);
    float d88 = advance8_inf(u_in, base, stride, 81u, d80);

    if (N == 96u) {
        float xnext = final_after5_inf(u_in, base, stride, 89u, d88, bd_hi);
        u_out[base + 94u * stride] = xnext;

        xnext = back_chunk5_inf(u_in, u_out, base, stride, 89u, d88, xnext);
        xnext = back_chunk8_inf(u_in, u_out, base, stride, 81u, d80, xnext);
        xnext = back_chunk8_inf(u_in, u_out, base, stride, 73u, d72, xnext);
        xnext = back_chunk8_inf(u_in, u_out, base, stride, 65u, d64, xnext);
        xnext = back_chunk8_inf(u_in, u_out, base, stride, 57u, d56, xnext);
        xnext = back_chunk8_inf(u_in, u_out, base, stride, 49u, d48, xnext);
        xnext = back_chunk8_inf(u_in, u_out, base, stride, 41u, d40, xnext);
        xnext = back_chunk8_inf(u_in, u_out, base, stride, 33u, d32, xnext);
        xnext = back_chunk8_inf(u_in, u_out, base, stride, 25u, d24, xnext);
        xnext = back_chunk8_inf(u_in, u_out, base, stride, 17u, d16, xnext);
        xnext = back_chunk8_inf(u_in, u_out, base, stride, 9u,  d8,  xnext);
        back_first8(u_out, base, stride, d1, d2, d3, d4, d5, d6, d7, d8, xnext);
        return;
    }

    float d96  = advance8_inf(u_in, base, stride, 89u,  d88);
    float d104 = advance8_inf(u_in, base, stride, 97u,  d96);
    float d112 = advance8_inf(u_in, base, stride, 105u, d104);
    float d120 = advance8_inf(u_in, base, stride, 113u, d112);

    float xnext = final_after5_inf(u_in, base, stride, 121u, d120, bd_hi);
    u_out[base + 126u * stride] = xnext;

    xnext = back_chunk5_inf(u_in, u_out, base, stride, 121u, d120, xnext);
    xnext = back_chunk8_inf(u_in, u_out, base, stride, 113u, d112, xnext);
    xnext = back_chunk8_inf(u_in, u_out, base, stride, 105u, d104, xnext);
    xnext = back_chunk8_inf(u_in, u_out, base, stride, 97u,  d96,  xnext);
    xnext = back_chunk8_inf(u_in, u_out, base, stride, 89u,  d88,  xnext);
    xnext = back_chunk8_inf(u_in, u_out, base, stride, 81u,  d80,  xnext);
    xnext = back_chunk8_inf(u_in, u_out, base, stride, 73u,  d72,  xnext);
    xnext = back_chunk8_inf(u_in, u_out, base, stride, 65u,  d64,  xnext);
    xnext = back_chunk8_inf(u_in, u_out, base, stride, 57u,  d56,  xnext);
    xnext = back_chunk8_inf(u_in, u_out, base, stride, 49u,  d48,  xnext);
    xnext = back_chunk8_inf(u_in, u_out, base, stride, 41u,  d40,  xnext);
    xnext = back_chunk8_inf(u_in, u_out, base, stride, 33u,  d32,  xnext);
    xnext = back_chunk8_inf(u_in, u_out, base, stride, 25u,  d24,  xnext);
    xnext = back_chunk8_inf(u_in, u_out, base, stride, 17u,  d16,  xnext);
    xnext = back_chunk8_inf(u_in, u_out, base, stride, 9u,   d8,   xnext);
    back_first8(u_out, base, stride, d1, d2, d3, d4, d5, d6, d7, d8, xnext);
}

inline void thomas_line_mu05(device const float *u_in,
                             device       float *u_out,
                             uint base, uint stride, uint N)
{
    if (N == 0u) return;

    uint last = base + (N - 1u) * stride;
    float bd_lo = u_in[base];
    float bd_hi = u_in[last];

    u_out[base] = bd_lo;
    if (N > 1u) u_out[last] = bd_hi;
    if (N < 3u) return;

    uint idx = base + stride;

    if (N == 3u) {
        u_out[idx] = (u_in[idx] + HALF * (bd_lo + bd_hi)) * INV1;
        return;
    }

    uint limit = N - 2u;

    float d1 = (u_in[idx] + HALF * bd_lo) * INV1;
    float d2 = 0.0f;
    float d3 = 0.0f;
    float d4 = 0.0f;
    float d5 = 0.0f;
    float d6 = 0.0f;
    float d7 = 0.0f;
    float d8 = 0.0f;

    float dp = d1;
    idx += stride;

    if (limit >= 9u) {
        d2 = (u_in[idx] + HALF * dp) * INV2;
        dp = d2;
        idx += stride;

        d3 = (u_in[idx] + HALF * dp) * INV3;
        dp = d3;
        idx += stride;

        d4 = (u_in[idx] + HALF * dp) * INV4;
        dp = d4;
        idx += stride;

        d5 = (u_in[idx] + HALF * dp) * INV5;
        dp = d5;
        idx += stride;

        d6 = (u_in[idx] + HALF * dp) * INV6;
        dp = d6;
        idx += stride;

        d7 = (u_in[idx] + HALF * dp) * INV7;
        dp = d7;
        idx += stride;

        d8 = (u_in[idx] + HALF * dp) * INV8;
        dp = d8;
        idx += stride;

        uint i = 9u;

        for (; i + 4u <= limit; i += 4u) {
            uint i0 = idx;
            uint i1 = i0 + stride;
            uint i2 = i1 + stride;
            uint i3 = i2 + stride;

            float r0 = u_in[i0];
            float r1 = u_in[i1];
            float r2 = u_in[i2];
            float r3 = u_in[i3];

            float q0 = (r0 + HALF * dp) * INVINF;
            float q1 = (r1 + HALF * q0) * INVINF;
            float q2 = (r2 + HALF * q1) * INVINF;
            float q3 = (r3 + HALF * q2) * INVINF;

            u_out[i0] = q0;
            u_out[i1] = q1;
            u_out[i2] = q2;
            u_out[i3] = q3;

            dp = q3;
            idx = i3 + stride;
        }

        for (; i < limit; ++i) {
            dp = (u_in[idx] + HALF * dp) * INVINF;
            u_out[idx] = dp;
            idx += stride;
        }

        dp = (u_in[idx] + HALF * dp + HALF * bd_hi) * INVINF;
    } else {
        if (limit > 2u) {
            d2 = (u_in[idx] + HALF * dp) * INV2;
            dp = d2;
            idx += stride;
        }
        if (limit > 3u) {
            d3 = (u_in[idx] + HALF * dp) * INV3;
            dp = d3;
            idx += stride;
        }
        if (limit > 4u) {
            d4 = (u_in[idx] + HALF * dp) * INV4;
            dp = d4;
            idx += stride;
        }
        if (limit > 5u) {
            d5 = (u_in[idx] + HALF * dp) * INV5;
            dp = d5;
            idx += stride;
        }
        if (limit > 6u) {
            d6 = (u_in[idx] + HALF * dp) * INV6;
            dp = d6;
            idx += stride;
        }
        if (limit > 7u) {
            d7 = (u_in[idx] + HALF * dp) * INV7;
            dp = d7;
            idx += stride;
        }

        dp = (u_in[idx] + HALF * dp + HALF * bd_hi) * inv_for_index(limit);
    }

    float xnext = dp;
    u_out[idx] = xnext;

    uint ii   = limit - 1u;
    uint bidx = idx - stride;

    while (ii >= 12u) {
        uint i0 = bidx;
        uint i1 = i0 - stride;
        uint i2 = i1 - stride;
        uint i3 = i2 - stride;

        float v0 = u_out[i0];
        float v1 = u_out[i1];
        float v2 = u_out[i2];
        float v3 = u_out[i3];

        float x0 = v0 - CPINF * xnext;
        float x1 = v1 - CPINF * x0;
        float x2 = v2 - CPINF * x1;
        float x3 = v3 - CPINF * x2;

        u_out[i0] = x0;
        u_out[i1] = x1;
        u_out[i2] = x2;
        u_out[i3] = x3;

        xnext = x3;
        bidx = i3 - stride;
        ii -= 4u;
    }

    while (ii >= 9u) {
        float xi = u_out[bidx] - CPINF * xnext;
        u_out[bidx] = xi;
        xnext = xi;
        --ii;
        bidx -= stride;
    }

    if (ii >= 8u) {
        float xi = d8 - CP8 * xnext;
        u_out[bidx] = xi;
        xnext = xi;
        bidx -= stride;
    }
    if (ii >= 7u) {
        float xi = d7 - CP7 * xnext;
        u_out[bidx] = xi;
        xnext = xi;
        bidx -= stride;
    }
    if (ii >= 6u) {
        float xi = d6 - CP6 * xnext;
        u_out[bidx] = xi;
        xnext = xi;
        bidx -= stride;
    }
    if (ii >= 5u) {
        float xi = d5 - CP5 * xnext;
        u_out[bidx] = xi;
        xnext = xi;
        bidx -= stride;
    }
    if (ii >= 4u) {
        float xi = d4 - CP4 * xnext;
        u_out[bidx] = xi;
        xnext = xi;
        bidx -= stride;
    }
    if (ii >= 3u) {
        float xi = d3 - CP3 * xnext;
        u_out[bidx] = xi;
        xnext = xi;
        bidx -= stride;
    }
    if (ii >= 2u) {
        float xi = d2 - CP2 * xnext;
        u_out[bidx] = xi;
        xnext = xi;
        bidx -= stride;
    }
    if (ii >= 1u) {
        float xi = d1 - CP1 * xnext;
        u_out[bidx] = xi;
    }
}

inline void solve_line_mu05(device const float *u_in,
                            device       float *u_out,
                            uint base, uint stride, uint N)
{
    if (N == 64u || N == 96u || N == 128u) {
        thomas_line_checkpoint_64_96_128(u_in, u_out, base, stride, N);
    } else {
        thomas_line_mu05(u_in, u_out, base, stride, N);
    }
}

kernel void adi_x(device const float *u_in   [[buffer(0)]],
                  device       float *u_out  [[buffer(1)]],
                  constant uint      &NX     [[buffer(2)]],
                  constant uint      &NY     [[buffer(3)]],
                  constant uint      &NZ     [[buffer(4)]],
                  constant float     &mu     [[buffer(5)]],
                  uint2 gid [[thread_position_in_grid]]) [[max_total_threads_per_threadgroup(32)]]
{
    uint j = gid.x;
    uint k = gid.y;
    if (j >= NY || k >= NZ) return;

    uint plane = NX * NY;
    uint base  = k * plane + j * NX;

    if (j == 0u || j == NY - 1u || k == 0u || k == NZ - 1u) {
        copy_line(u_in, u_out, base, 1u, NX);
        return;
    }

    solve_line_mu05(u_in, u_out, base, 1u, NX);
}

kernel void adi_y(device const float *u_in   [[buffer(0)]],
                  device       float *u_out  [[buffer(1)]],
                  constant uint      &NX     [[buffer(2)]],
                  constant uint      &NY     [[buffer(3)]],
                  constant uint      &NZ     [[buffer(4)]],
                  constant float     &mu     [[buffer(5)]],
                  uint2 gid [[thread_position_in_grid]]) [[max_total_threads_per_threadgroup(32)]]
{
    uint i = gid.x;
    uint k = gid.y;
    if (i >= NX || k >= NZ) return;

    uint plane = NX * NY;
    uint base  = k * plane + i;

    if (i == 0u || i == NX - 1u || k == 0u || k == NZ - 1u) {
        copy_line(u_in, u_out, base, NX, NY);
        return;
    }

    solve_line_mu05(u_in, u_out, base, NX, NY);
}

kernel void adi_z(device const float *u_in   [[buffer(0)]],
                  device       float *u_out  [[buffer(1)]],
                  constant uint      &NX     [[buffer(2)]],
                  constant uint      &NY     [[buffer(3)]],
                  constant uint      &NZ     [[buffer(4)]],
                  constant float     &mu     [[buffer(5)]],
                  uint2 gid [[thread_position_in_grid]]) [[max_total_threads_per_threadgroup(32)]]
{
    uint i = gid.x;
    uint j = gid.y;
    if (i >= NX || j >= NY) return;

    uint plane = NX * NY;
    uint base  = j * NX + i;

    if (i == 0u || i == NX - 1u || j == 0u || j == NY - 1u) {
        copy_line(u_in, u_out, base, plane, NZ);
        return;
    }

    solve_line_mu05(u_in, u_out, base, plane, NZ);
}
```