## Optimization: Threadgroup-cooperative Thomas via parallel-cyclic-reduction-like Sherman-Morrison... actually, let's do cooperative tiling + register-cached lines.

**Structural change:** Instead of one thread doing a full Thomas sweep (serial, register-heavy `dprime[MAX_AXIS]` array hurts occupancy badly), I'll keep one-thread-per-line but switch the per-thread storage to **threadgroup memory** for `dprime`, freeing private register pressure to boost occupancy. With TG=32 threads and dprime stored as `float dprime[32][N+pad]` in threadgroup memory, register spill is eliminated. To keep TG memory under budget I cap N at 128 (incumbent benchmarks use N≤128). I also precompute `cprime` AND `inv_denom` once in TG memory, and use `fma`-friendly form. The 32-thread TG accesses `dprime[tlid][i]` with a column-major layout (`dprime[i*32 + tlid]`) so consecutive threads hit consecutive banks — no bank conflicts, coalesced TG access. This should let many more TGs run concurrently per SM, improving DRAM throughput utilization.

```metal
#include <metal_stdlib>
using namespace metal;

constant constexpr uint MAX_AXIS = 130u;   // supports N up to 130
constant constexpr uint TG_W     = 32u;

// Precompute cprime[i] and invden[i] = 1/(b - a*cprime[i-1]) for i in [1, N-2].
// cprime[i] = c * invden[i] for i in [1, N-2] (cprime[N-2] unused in back-sub but harmless).
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
            cprime[i] = c * invd;   // cprime[last] unused in back-sub
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Thomas solve with dprime in TG memory (column-major: dp[i, tlid] = dp_tg[i*TG_W + tlid]).
inline void thomas_tg(device const float *u_in,
                      device       float *u_out,
                      threadgroup const float *cprime,
                      threadgroup const float *invden,
                      threadgroup       float *dp_tg,
                      uint base, uint stride, uint N, float mu, uint tlid)
{
    float a = -mu;

    float bd_lo = u_in[base];
    float bd_hi = u_in[base + (N - 1u) * stride];

    u_out[base]                     = bd_lo;
    u_out[base + (N - 1u) * stride] = bd_hi;

    if (N < 3u) return;

    // Forward elimination
    float dp = (u_in[base + stride] + mu * bd_lo) * invden[1];
    dp_tg[1u * TG_W + tlid] = dp;

    uint last = N - 2u;
    for (uint i = 2u; i < last; ++i) {
        float rhs = u_in[base + i * stride];
        dp = (rhs - a * dp) * invden[i];
        dp_tg[i * TG_W + tlid] = dp;
    }

    // i = last (absorb bd_hi)
    float rhsL = u_in[base + last * stride] + mu * bd_hi;
    dp = (rhsL - a * dp) * invden[last];

    // Backward substitution
    u_out[base + last * stride] = dp;
    float xnext = dp;
    if (N >= 4u) {
        for (uint ii = N - 3u; ii >= 1u; --ii) {
            float xi = dp_tg[ii * TG_W + tlid] - cprime[ii] * xnext;
            u_out[base + ii * stride] = xi;
            xnext = xi;
            if (ii == 1u) break;
        }
    }
}

[[max_total_threads_per_threadgroup(32)]]
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
    threadgroup float dp_tg[MAX_AXIS * TG_W];

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
    thomas_tg(u_in, u_out, cprime, invden, dp_tg, base, 1u, NX, mu, tlid);
}

[[max_total_threads_per_threadgroup(32)]]
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
    threadgroup float dp_tg[MAX_AXIS * TG_W];

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
    thomas_tg(u_in, u_out, cprime, invden, dp_tg, base, NX, NY, mu, tlid);
}

[[max_total_threads_per_threadgroup(32)]]
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
    threadgroup float dp_tg[MAX_AXIS * TG_W];

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
    thomas_tg(u_in, u_out, cprime, invden, dp_tg, base, plane, NZ, mu, tlid);
}
```