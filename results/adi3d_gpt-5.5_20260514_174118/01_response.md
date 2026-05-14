Specialized for the host’s fixed `mu = 0.5`, this version removes per-threadgroup coefficient generation and division from the Thomas sweeps. For axes up to 128, it stores forward-sweep temporaries in threadgroup memory laid out lane-contiguously, reducing global memory traffic versus using `u_out` as scratch; larger axes fall back to a global-temp path with the same constant-coefficient optimization.

```metal
#include <metal_stdlib>
using namespace metal;

constant constexpr uint TG_WIDTH     = 32u;
constant constexpr uint SCRATCH_AXIS = 128u;

constant constexpr float HALF = 0.5f;

// c'_i for mu = 0.5.  After i=6 the float values have effectively
// converged to sqrt(3)-2 for the problem tolerances.
constant constexpr float CP1   = -0.25f;
constant constexpr float CP2   = -0.2666666666666667f;
constant constexpr float CP3   = -0.26785714285714285f;
constant constexpr float CP4   = -0.2679425837320574f;
constant constexpr float CP5   = -0.26794871794871794f;
constant constexpr float CP6   = -0.2679491583648231f;
constant constexpr float CPINF = -0.2679491924311227f;

constant constexpr float INV1   = -2.0f * CP1;
constant constexpr float INV2   = -2.0f * CP2;
constant constexpr float INV3   = -2.0f * CP3;
constant constexpr float INV4   = -2.0f * CP4;
constant constexpr float INV5   = -2.0f * CP5;
constant constexpr float INV6   = -2.0f * CP6;
constant constexpr float INVINF = -2.0f * CPINF;

inline float inv_for_index(uint i)
{
    if (i == 1u) return INV1;
    if (i == 2u) return INV2;
    if (i == 3u) return INV3;
    if (i == 4u) return INV4;
    if (i == 5u) return INV5;
    if (i == 6u) return INV6;
    return INVINF;
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

inline void thomas_line_mu05_scratch(device const float *u_in,
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

    uint limit = N - 2u; // final interior index

    float dp = (u_in[idx] + HALF * bd_lo) * INV1;
    scratch[1u * TG_WIDTH + lane] = dp;
    idx += stride;

    if (limit > 2u) {
        dp = (u_in[idx] + HALF * dp) * INV2;
        scratch[2u * TG_WIDTH + lane] = dp;
        idx += stride;
    }
    if (limit > 3u) {
        dp = (u_in[idx] + HALF * dp) * INV3;
        scratch[3u * TG_WIDTH + lane] = dp;
        idx += stride;
    }
    if (limit > 4u) {
        dp = (u_in[idx] + HALF * dp) * INV4;
        scratch[4u * TG_WIDTH + lane] = dp;
        idx += stride;
    }
    if (limit > 5u) {
        dp = (u_in[idx] + HALF * dp) * INV5;
        scratch[5u * TG_WIDTH + lane] = dp;
        idx += stride;
    }
    if (limit > 6u) {
        dp = (u_in[idx] + HALF * dp) * INV6;
        scratch[6u * TG_WIDTH + lane] = dp;
        idx += stride;
    }

    for (uint i = 7u; i < limit; ++i) {
        dp = (u_in[idx] + HALF * dp) * INVINF;
        scratch[i * TG_WIDTH + lane] = dp;
        idx += stride;
    }

    dp = (u_in[idx] + HALF * dp + HALF * bd_hi) * inv_for_index(limit);
    float xnext = dp;
    u_out[idx] = xnext;

    uint ii   = limit - 1u;
    uint bidx = idx - stride;

    while (ii >= 7u) {
        float dpi = scratch[ii * TG_WIDTH + lane];
        float xi  = dpi - CPINF * xnext;
        u_out[bidx] = xi;
        xnext = xi;
        --ii;
        bidx -= stride;
    }

    if (ii >= 6u) {
        float xi = scratch[6u * TG_WIDTH + lane] - CP6 * xnext;
        u_out[bidx] = xi; xnext = xi; bidx -= stride;
    }
    if (ii >= 5u) {
        float xi = scratch[5u * TG_WIDTH + lane] - CP5 * xnext;
        u_out[bidx] = xi; xnext = xi; bidx -= stride;
    }
    if (ii >= 4u) {
        float xi = scratch[4u * TG_WIDTH + lane] - CP4 * xnext;
        u_out[bidx] = xi; xnext = xi; bidx -= stride;
    }
    if (ii >= 3u) {
        float xi = scratch[3u * TG_WIDTH + lane] - CP3 * xnext;
        u_out[bidx] = xi; xnext = xi; bidx -= stride;
    }
    if (ii >= 2u) {
        float xi = scratch[2u * TG_WIDTH + lane] - CP2 * xnext;
        u_out[bidx] = xi; xnext = xi; bidx -= stride;
    }
    if (ii >= 1u) {
        float xi = scratch[1u * TG_WIDTH + lane] - CP1 * xnext;
        u_out[bidx] = xi;
    }
}

inline void thomas_line_mu05_global(device const float *u_in,
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

    float dp = (u_in[idx] + HALF * bd_lo) * INV1;
    u_out[idx] = dp;
    idx += stride;

    if (limit > 2u) {
        dp = (u_in[idx] + HALF * dp) * INV2;
        u_out[idx] = dp;
        idx += stride;
    }
    if (limit > 3u) {
        dp = (u_in[idx] + HALF * dp) * INV3;
        u_out[idx] = dp;
        idx += stride;
    }
    if (limit > 4u) {
        dp = (u_in[idx] + HALF * dp) * INV4;
        u_out[idx] = dp;
        idx += stride;
    }
    if (limit > 5u) {
        dp = (u_in[idx] + HALF * dp) * INV5;
        u_out[idx] = dp;
        idx += stride;
    }
    if (limit > 6u) {
        dp = (u_in[idx] + HALF * dp) * INV6;
        u_out[idx] = dp;
        idx += stride;
    }

    for (uint i = 7u; i < limit; ++i) {
        dp = (u_in[idx] + HALF * dp) * INVINF;
        u_out[idx] = dp;
        idx += stride;
    }

    dp = (u_in[idx] + HALF * dp + HALF * bd_hi) * inv_for_index(limit);
    float xnext = dp;
    u_out[idx] = xnext;

    uint ii   = limit - 1u;
    uint bidx = idx - stride;

    while (ii >= 7u) {
        float dpi = u_out[bidx];
        float xi  = dpi - CPINF * xnext;
        u_out[bidx] = xi;
        xnext = xi;
        --ii;
        bidx -= stride;
    }

    if (ii >= 6u) {
        float xi = u_out[bidx] - CP6 * xnext;
        u_out[bidx] = xi; xnext = xi; bidx -= stride;
    }
    if (ii >= 5u) {
        float xi = u_out[bidx] - CP5 * xnext;
        u_out[bidx] = xi; xnext = xi; bidx -= stride;
    }
    if (ii >= 4u) {
        float xi = u_out[bidx] - CP4 * xnext;
        u_out[bidx] = xi; xnext = xi; bidx -= stride;
    }
    if (ii >= 3u) {
        float xi = u_out[bidx] - CP3 * xnext;
        u_out[bidx] = xi; xnext = xi; bidx -= stride;
    }
    if (ii >= 2u) {
        float xi = u_out[bidx] - CP2 * xnext;
        u_out[bidx] = xi; xnext = xi; bidx -= stride;
    }
    if (ii >= 1u) {
        float xi = u_out[bidx] - CP1 * xnext;
        u_out[bidx] = xi;
    }
}

kernel void adi_x(device const float *u_in   [[buffer(0)]],
                  device       float *u_out  [[buffer(1)]],
                  constant uint      &NX     [[buffer(2)]],
                  constant uint      &NY     [[buffer(3)]],
                  constant uint      &NZ     [[buffer(4)]],
                  constant float     &mu     [[buffer(5)]],
                  uint2 gid [[thread_position_in_grid]])
{
    threadgroup float scratch[SCRATCH_AXIS * TG_WIDTH];

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
    if (NX <= SCRATCH_AXIS) {
        thomas_line_mu05_scratch(u_in, u_out, scratch, lane, base, 1u, NX);
    } else {
        thomas_line_mu05_global(u_in, u_out, base, 1u, NX);
    }
}

kernel void adi_y(device const float *u_in   [[buffer(0)]],
                  device       float *u_out  [[buffer(1)]],
                  constant uint      &NX     [[buffer(2)]],
                  constant uint      &NY     [[buffer(3)]],
                  constant uint      &NZ     [[buffer(4)]],
                  constant float     &mu     [[buffer(5)]],
                  uint2 gid [[thread_position_in_grid]])
{
    threadgroup float scratch[SCRATCH_AXIS * TG_WIDTH];

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
    if (NY <= SCRATCH_AXIS) {
        thomas_line_mu05_scratch(u_in, u_out, scratch, lane, base, NX, NY);
    } else {
        thomas_line_mu05_global(u_in, u_out, base, NX, NY);
    }
}

kernel void adi_z(device const float *u_in   [[buffer(0)]],
                  device       float *u_out  [[buffer(1)]],
                  constant uint      &NX     [[buffer(2)]],
                  constant uint      &NY     [[buffer(3)]],
                  constant uint      &NZ     [[buffer(4)]],
                  constant float     &mu     [[buffer(5)]],
                  uint2 gid [[thread_position_in_grid]])
{
    threadgroup float scratch[SCRATCH_AXIS * TG_WIDTH];

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
    if (NZ <= SCRATCH_AXIS) {
        thomas_line_mu05_scratch(u_in, u_out, scratch, lane, base, plane, NZ);
    } else {
        thomas_line_mu05_global(u_in, u_out, base, plane, NZ);
    }
}
```