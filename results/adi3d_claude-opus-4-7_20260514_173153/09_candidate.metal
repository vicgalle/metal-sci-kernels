#include <metal_stdlib>
using namespace metal;

constant constexpr uint MAX_AXIS = 144u;
constant constexpr uint TG_W     = 32u;

inline void init_cprime(threadgroup float *cprime,
                        uint N, float mu, uint tlid)
{
    if (tlid == 0u && N >= 3u) {
        float a = -mu;
        float b = 1.0f + 2.0f * mu;
        float c = -mu;
        float prev = c / b;
        cprime[1] = prev;
        for (uint i = 2u; i < N - 1u; ++i) {
            prev = c / (b - a * prev);
            cprime[i] = prev;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Thomas solve with dprime in threadgroup memory (one row per lane).
inline void thomas_line_tg(device const float *u_in,
                           device       float *u_out,
                           threadgroup const float *cprime,
                           threadgroup float *dprime_row,
                           uint base, uint stride, uint N, float mu)
{
    float a = -mu;
    float b = 1.0f + 2.0f * mu;

    float bd_lo = u_in[base];
    float bd_hi = u_in[base + (N - 1u) * stride];

    u_out[base]                     = bd_lo;
    u_out[base + (N - 1u) * stride] = bd_hi;

    if (N < 3u) return;

    // i = 1
    float dp = (u_in[base + stride] + mu * bd_lo) / b;
    dprime_row[1 * TG_W] = dp;

    // i = 2 .. N - 3
    uint Nm2 = N - 2u;
    for (uint i = 2u; i < Nm2; ++i) {
        float denom = b - a * cprime[i - 1u];
        dp = (u_in[base + i * stride] - a * dp) / denom;
        dprime_row[i * TG_W] = dp;
    }

    // i = N - 2 (absorb upper boundary)
    {
        float denom = b - a * cprime[Nm2 - 1u];
        dp = ((u_in[base + Nm2 * stride] + mu * bd_hi) - a * dp) / denom;
        u_out[base + Nm2 * stride] = dp;
    }

    // Backward substitution
    float xnext = dp;
    for (uint ii = N - 3u; ii >= 1u; --ii) {
        float xi = dprime_row[ii * TG_W] - cprime[ii] * xnext;
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
                  uint  tlid [[thread_index_in_threadgroup]]) [[max_total_threads_per_threadgroup(32)]]
{
    threadgroup float cprime[MAX_AXIS];
    threadgroup float dprime[MAX_AXIS * TG_W];
    init_cprime(cprime, NX, mu, tlid);

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
    thomas_line_tg(u_in, u_out, cprime, dprime + tlid, base, 1u, NX, mu);
}

kernel void adi_y(device const float *u_in   [[buffer(0)]],
                  device       float *u_out  [[buffer(1)]],
                  constant uint      &NX     [[buffer(2)]],
                  constant uint      &NY     [[buffer(3)]],
                  constant uint      &NZ     [[buffer(4)]],
                  constant float     &mu     [[buffer(5)]],
                  uint2 gid  [[thread_position_in_grid]],
                  uint  tlid [[thread_index_in_threadgroup]]) [[max_total_threads_per_threadgroup(32)]]
{
    threadgroup float cprime[MAX_AXIS];
    threadgroup float dprime[MAX_AXIS * TG_W];
    init_cprime(cprime, NY, mu, tlid);

    uint i = gid.x;
    uint k = gid.y;
    if (i >= NX || k >= NZ) return;

    uint plane = NX * NY;
    uint base  = k * plane + i;

    if (i == 0u || i == NX - 1u || k == 0u || k == NZ - 1u) {
        for (uint j = 0u; j < NY; ++j) u_out[base + j * NX] = u_in[base + j * NX];
        return;
    }
    thomas_line_tg(u_in, u_out, cprime, dprime + tlid, base, NX, NY, mu);
}

kernel void adi_z(device const float *u_in   [[buffer(0)]],
                  device       float *u_out  [[buffer(1)]],
                  constant uint      &NX     [[buffer(2)]],
                  constant uint      &NY     [[buffer(3)]],
                  constant uint      &NZ     [[buffer(4)]],
                  constant float     &mu     [[buffer(5)]],
                  uint2 gid  [[thread_position_in_grid]],
                  uint  tlid [[thread_index_in_threadgroup]]) [[max_total_threads_per_threadgroup(32)]]
{
    threadgroup float cprime[MAX_AXIS];
    threadgroup float dprime[MAX_AXIS * TG_W];
    init_cprime(cprime, NZ, mu, tlid);

    uint i = gid.x;
    uint j = gid.y;
    if (i >= NX || j >= NY) return;

    uint plane = NX * NY;
    uint base  = j * NX + i;

    if (i == 0u || i == NX - 1u || j == 0u || j == NY - 1u) {
        for (uint k = 0u; k < NZ; ++k) u_out[base + k * plane] = u_in[base + k * plane];
        return;
    }
    thomas_line_tg(u_in, u_out, cprime, dprime + tlid, base, plane, NZ, mu);
}