#include <metal_stdlib>
using namespace metal;

constant constexpr uint  SIMD_W    = 32u;
constant constexpr uint  SCRATCH_N = 64u;
constant constexpr float HALF      = 0.5f;

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
    for (uint n = 0u; n < N; ++n) {
        u_out[idx] = u_in[idx];
        idx += stride;
    }
}

inline void thomas_line_mu05_cached64(device const float *u_in,
                                      device       float *u_out,
                                      threadgroup  float *scratch,
                                      uint lane,
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

    uint limit = N - 2u; // final interior index, >= 2 here

    // Forward elimination.  Store d'_1..d'_64 in threadgroup memory;
    // longer-line tail d' values use u_out as scratch.
    float dp = (u_in[idx] + HALF * bd_lo) * INV1;
    scratch[lane] = dp; // i = 1
    idx += stride;

    if (limit > 2u) {
        dp = (u_in[idx] + HALF * dp) * INV2;
        scratch[1u * SIMD_W + lane] = dp;
        idx += stride;
    }
    if (limit > 3u) {
        dp = (u_in[idx] + HALF * dp) * INV3;
        scratch[2u * SIMD_W + lane] = dp;
        idx += stride;
    }
    if (limit > 4u) {
        dp = (u_in[idx] + HALF * dp) * INV4;
        scratch[3u * SIMD_W + lane] = dp;
        idx += stride;
    }
    if (limit > 5u) {
        dp = (u_in[idx] + HALF * dp) * INV5;
        scratch[4u * SIMD_W + lane] = dp;
        idx += stride;
    }
    if (limit > 6u) {
        dp = (u_in[idx] + HALF * dp) * INV6;
        scratch[5u * SIMD_W + lane] = dp;
        idx += stride;
    }
    if (limit > 7u) {
        dp = (u_in[idx] + HALF * dp) * INV7;
        scratch[6u * SIMD_W + lane] = dp;
        idx += stride;
    }
    if (limit > 8u) {
        dp = (u_in[idx] + HALF * dp) * INV8;
        scratch[7u * SIMD_W + lane] = dp;
        idx += stride;
    }

    uint i = 9u;
    for (; i < limit && i <= SCRATCH_N; ++i) {
        dp = (u_in[idx] + HALF * dp) * INVINF;
        scratch[(i - 1u) * SIMD_W + lane] = dp;
        idx += stride;
    }

    for (; i < limit; ++i) {
        dp = (u_in[idx] + HALF * dp) * INVINF;
        u_out[idx] = dp;
        idx += stride;
    }

    // Final interior unknown includes the high Dirichlet endpoint.
    dp = (u_in[idx] + HALF * dp + HALF * bd_hi) * inv_for_index(limit);
    float xnext = dp;
    u_out[idx] = xnext;

    // Back substitution.  Tail d' values, if any, are in u_out and are hot;
    // first SCRATCH_N d' values are read from threadgroup memory.
    uint ii   = limit - 1u;
    uint bidx = idx - stride;

    while (ii > SCRATCH_N) {
        float xi = u_out[bidx] - CPINF * xnext;
        u_out[bidx] = xi;
        xnext = xi;
        --ii;
        bidx -= stride;
    }

    while (ii >= 9u) {
        float xi = scratch[(ii - 1u) * SIMD_W + lane] - CPINF * xnext;
        u_out[bidx] = xi;
        xnext = xi;
        --ii;
        bidx -= stride;
    }

    if (ii >= 8u) {
        float xi = scratch[7u * SIMD_W + lane] - CP8 * xnext;
        u_out[bidx] = xi; xnext = xi; bidx -= stride;
    }
    if (ii >= 7u) {
        float xi = scratch[6u * SIMD_W + lane] - CP7 * xnext;
        u_out[bidx] = xi; xnext = xi; bidx -= stride;
    }
    if (ii >= 6u) {
        float xi = scratch[5u * SIMD_W + lane] - CP6 * xnext;
        u_out[bidx] = xi; xnext = xi; bidx -= stride;
    }
    if (ii >= 5u) {
        float xi = scratch[4u * SIMD_W + lane] - CP5 * xnext;
        u_out[bidx] = xi; xnext = xi; bidx -= stride;
    }
    if (ii >= 4u) {
        float xi = scratch[3u * SIMD_W + lane] - CP4 * xnext;
        u_out[bidx] = xi; xnext = xi; bidx -= stride;
    }
    if (ii >= 3u) {
        float xi = scratch[2u * SIMD_W + lane] - CP3 * xnext;
        u_out[bidx] = xi; xnext = xi; bidx -= stride;
    }
    if (ii >= 2u) {
        float xi = scratch[1u * SIMD_W + lane] - CP2 * xnext;
        u_out[bidx] = xi; xnext = xi; bidx -= stride;
    }
    if (ii >= 1u) {
        float xi = scratch[lane] - CP1 * xnext;
        u_out[bidx] = xi;
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
    threadgroup float scratch[SCRATCH_N * SIMD_W];

    uint j = gid.x;
    uint k = gid.y;
    if (j >= NY || k >= NZ) return;

    uint plane = NX * NY;
    uint base  = k * plane + j * NX;

    if (j == 0u || j == NY - 1u || k == 0u || k == NZ - 1u) {
        copy_line(u_in, u_out, base, 1u, NX);
        return;
    }

    uint lane = j & 31u;
    thomas_line_mu05_cached64(u_in, u_out, scratch, lane, base, 1u, NX);
}

kernel void adi_y(device const float *u_in   [[buffer(0)]],
                  device       float *u_out  [[buffer(1)]],
                  constant uint      &NX     [[buffer(2)]],
                  constant uint      &NY     [[buffer(3)]],
                  constant uint      &NZ     [[buffer(4)]],
                  constant float     &mu     [[buffer(5)]],
                  uint2 gid [[thread_position_in_grid]]) [[max_total_threads_per_threadgroup(32)]]
{
    threadgroup float scratch[SCRATCH_N * SIMD_W];

    uint i = gid.x;
    uint k = gid.y;
    if (i >= NX || k >= NZ) return;

    uint plane = NX * NY;
    uint base  = k * plane + i;

    if (i == 0u || i == NX - 1u || k == 0u || k == NZ - 1u) {
        copy_line(u_in, u_out, base, NX, NY);
        return;
    }

    uint lane = i & 31u;
    thomas_line_mu05_cached64(u_in, u_out, scratch, lane, base, NX, NY);
}

kernel void adi_z(device const float *u_in   [[buffer(0)]],
                  device       float *u_out  [[buffer(1)]],
                  constant uint      &NX     [[buffer(2)]],
                  constant uint      &NY     [[buffer(3)]],
                  constant uint      &NZ     [[buffer(4)]],
                  constant float     &mu     [[buffer(5)]],
                  uint2 gid [[thread_position_in_grid]]) [[max_total_threads_per_threadgroup(32)]]
{
    threadgroup float scratch[SCRATCH_N * SIMD_W];

    uint i = gid.x;
    uint j = gid.y;
    if (i >= NX || j >= NY) return;

    uint plane = NX * NY;
    uint base  = j * NX + i;

    if (i == 0u || i == NX - 1u || j == 0u || j == NY - 1u) {
        copy_line(u_in, u_out, base, plane, NZ);
        return;
    }

    uint lane = i & 31u;
    thomas_line_mu05_cached64(u_in, u_out, scratch, lane, base, plane, NZ);
}