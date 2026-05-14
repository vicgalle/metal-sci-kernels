#include <metal_stdlib>
using namespace metal;

constant constexpr uint MAX_AXIS = 130u;

inline void init_tables(threadgroup float *cprime,
                        threadgroup float *invden,
                        uint N, float mu, uint tlid)
{
    if (tlid == 0u && N >= 3u) {
        float a = -mu;
        float b = 1.0f + 2.0f * mu;
        float c = -mu;
        float inv = 1.0f / b;
        invden[1] = inv;
        cprime[1] = c * inv;
        uint last = N - 2u;
        for (uint i = 2u; i <= last; ++i) {
            float denom = b - a * cprime[i - 1u];
            float invd  = 1.0f / denom;
            invden[i] = invd;
            cprime[i] = c * invd;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Thomas solve using u_out itself to hold d' during forward elimination.
// Each interior cell of u_out is written once with d'[i], then overwritten
// during back-substitution with the final solution.
inline void thomas_inplace(device const float *u_in,
                           device       float *u_out,
                           threadgroup const float *cprime,
                           threadgroup const float *invden,
                           uint base, uint stride, uint N, float mu)
{
    float a = -mu;

    float bd_lo = u_in[base];
    float bd_hi = u_in[base + (N - 1u) * stride];

    u_out[base]                     = bd_lo;
    u_out[base + (N - 1u) * stride] = bd_hi;

    if (N < 3u) return;

    uint last = N - 2u;

    // i = 1 (absorb low boundary)
    float dp = (u_in[base + stride] + mu * bd_lo) * invden[1];
    u_out[base + stride] = dp;

    // i = 2 .. last-1
    for (uint i = 2u; i < last; ++i) {
        float rhs = u_in[base + i * stride];
        dp = (rhs - a * dp) * invden[i];
        u_out[base + i * stride] = dp;
    }

    // i = last (absorb high boundary): this is the last d', and also
    // the solution x[last] since back-sub starts here.
    float rhsL = u_in[base + last * stride] + mu * bd_hi;
    dp = (rhsL - a * dp) * invden[last];
    u_out[base + last * stride] = dp;

    // Backward substitution: read d' from u_out, overwrite with x.
    float xnext = dp;
    for (uint ii = N - 3u; ii >= 1u; --ii) {
        float di = u_out[base + ii * stride];
        float xi = di - cprime[ii] * xnext;
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
                  uint2 gid  [[thread_position_in_grid]],
                  uint  tlid [[thread_index_in_threadgroup]])
{
    threadgroup float cprime[MAX_AXIS];
    threadgroup float invden[MAX_AXIS];
    init_tables(cprime, invden, NX, mu, tlid);

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
    thomas_inplace(u_in, u_out, cprime, invden, base, 1u, NX, mu);
}

kernel void adi_y(device const float *u_in   [[buffer(0)]],
                  device       float *u_out  [[buffer(1)]],
                  constant uint      &NX     [[buffer(2)]],
                  constant uint      &NY     [[buffer(3)]],
                  constant uint      &NZ     [[buffer(4)]],
                  constant float     &mu     [[buffer(5)]],
                  uint2 gid  [[thread_position_in_grid]],
                  uint  tlid [[thread_index_in_threadgroup]])
{
    threadgroup float cprime[MAX_AXIS];
    threadgroup float invden[MAX_AXIS];
    init_tables(cprime, invden, NY, mu, tlid);

    uint i = gid.x;
    uint k = gid.y;
    if (i >= NX || k >= NZ) return;

    uint plane = NX * NY;
    uint base  = k * plane + i;

    if (i == 0u || i == NX - 1u || k == 0u || k == NZ - 1u) {
        for (uint j = 0u; j < NY; ++j) u_out[base + j * NX] = u_in[base + j * NX];
        return;
    }
    thomas_inplace(u_in, u_out, cprime, invden, base, NX, NY, mu);
}

kernel void adi_z(device const float *u_in   [[buffer(0)]],
                  device       float *u_out  [[buffer(1)]],
                  constant uint      &NX     [[buffer(2)]],
                  constant uint      &NY     [[buffer(3)]],
                  constant uint      &NZ     [[buffer(4)]],
                  constant float     &mu     [[buffer(5)]],
                  uint2 gid  [[thread_position_in_grid]],
                  uint  tlid [[thread_index_in_threadgroup]])
{
    threadgroup float cprime[MAX_AXIS];
    threadgroup float invden[MAX_AXIS];
    init_tables(cprime, invden, NZ, mu, tlid);

    uint i = gid.x;
    uint j = gid.y;
    if (i >= NX || j >= NY) return;

    uint plane = NX * NY;
    uint base  = j * NX + i;

    if (i == 0u || i == NX - 1u || j == 0u || j == NY - 1u) {
        for (uint k = 0u; k < NZ; ++k) u_out[base + k * plane] = u_in[base + k * plane];
        return;
    }
    thomas_inplace(u_in, u_out, cprime, invden, base, plane, NZ, mu);
}