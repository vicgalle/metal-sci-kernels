#include <metal_stdlib>
using namespace metal;

constant constexpr uint MAX_AXIS = 256u;

inline void thomas_line(device const float *u_in,
                        device       float *u_out,
                        uint base, uint stride, uint N, float mu)
{
    float a = -mu;
    float b = 1.0f + 2.0f * mu;

    float bd_lo = u_in[base];
    float bd_hi = u_in[base + (N - 1u) * stride];

    u_out[base]                     = bd_lo;
    u_out[base + (N - 1u) * stride] = bd_hi;

    if (N < 3u) return;

    thread float cprime[MAX_AXIS];
    thread float dprime[MAX_AXIS];

    // Forward sweep: compute c' and d' together
    float inv_b = 1.0f / b;
    float cp = (-a) * inv_b;  // c' = c/b, but c = -mu = a, so cp = -a/b... wait c = -mu = a
    // Actually a = c = -mu, both equal. c' = c/b = -mu/b
    cp = (-mu) * inv_b;
    cprime[1] = cp;
    float dp = (u_in[base + stride] + mu * bd_lo) * inv_b;
    dprime[1] = dp;

    // i = 2 .. N - 3
    for (uint i = 2u; i + 1u < N - 1u; ++i) {
        float inv = 1.0f / (b - a * cp);
        cp = (-mu) * inv;
        cprime[i] = cp;
        dp = (u_in[base + i * stride] - a * dp) * inv;
        dprime[i] = dp;
    }

    // i = N - 2
    uint iN = N - 2u;
    float inv = 1.0f / (b - a * cp);
    dp = ((u_in[base + iN * stride] + mu * bd_hi) - a * dp) * inv;
    u_out[base + iN * stride] = dp;

    // Backward sub
    float xnext = dp;
    for (uint ii = N - 3u; ii >= 1u; --ii) {
        float xi = dprime[ii] - cprime[ii] * xnext;
        u_out[base + ii * stride] = xi;
        xnext = xi;
        if (ii == 1u) break;
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
    uint j = gid.x;
    uint k = gid.y;
    if (j >= NY || k >= NZ) return;

    uint plane = NX * NY;
    uint base  = k * plane + j * NX;

    if (j == 0u || j == NY - 1u || k == 0u || k == NZ - 1u) {
        uint i = 0u;
        for (; i + 4u <= NX; i += 4u) {
            float4 v = *reinterpret_cast<device const float4*>(&u_in[base + i]);
            *reinterpret_cast<device float4*>(&u_out[base + i]) = v;
        }
        for (; i < NX; ++i) u_out[base + i] = u_in[base + i];
        return;
    }
    thomas_line(u_in, u_out, base, 1u, NX, mu);
}

kernel void adi_y(device const float *u_in   [[buffer(0)]],
                  device       float *u_out  [[buffer(1)]],
                  constant uint      &NX     [[buffer(2)]],
                  constant uint      &NY     [[buffer(3)]],
                  constant uint      &NZ     [[buffer(4)]],
                  constant float     &mu     [[buffer(5)]],
                  uint2 gid [[thread_position_in_grid]])
{
    uint i = gid.x;
    uint k = gid.y;
    if (i >= NX || k >= NZ) return;

    uint plane = NX * NY;
    uint base  = k * plane + i;

    if (i == 0u || i == NX - 1u || k == 0u || k == NZ - 1u) {
        for (uint j = 0u; j < NY; ++j) u_out[base + j * NX] = u_in[base + j * NX];
        return;
    }
    thomas_line(u_in, u_out, base, NX, NY, mu);
}

kernel void adi_z(device const float *u_in   [[buffer(0)]],
                  device       float *u_out  [[buffer(1)]],
                  constant uint      &NX     [[buffer(2)]],
                  constant uint      &NY     [[buffer(3)]],
                  constant uint      &NZ     [[buffer(4)]],
                  constant float     &mu     [[buffer(5)]],
                  uint2 gid [[thread_position_in_grid]])
{
    uint i = gid.x;
    uint j = gid.y;
    if (i >= NX || j >= NY) return;

    uint plane = NX * NY;
    uint base  = j * NX + i;

    if (i == 0u || i == NX - 1u || j == 0u || j == NY - 1u) {
        for (uint k = 0u; k < NZ; ++k) u_out[base + k * plane] = u_in[base + k * plane];
        return;
    }
    thomas_line(u_in, u_out, base, plane, NZ, mu);
}