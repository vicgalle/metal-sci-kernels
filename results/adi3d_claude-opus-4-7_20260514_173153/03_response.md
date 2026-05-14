**Optimization**: The main bottleneck is that adi_y and adi_z have strided memory accesses (stride NX and NX*NY) that scatter cache lines. Since the threadgroup is (32,1,1) along gid.x, in adi_y consecutive threads share the same k and have consecutive i, so their reads at each step `base + j*NX + i` are contiguous in memory — already coalesced. Same for adi_z (consecutive i, same j). So the access pattern is actually friendly. The remaining wins: (1) precompute `cprime` once into threadgroup memory in parallel by all 32 threads cooperating (cheap, but the recurrence is serial — keep serial but only do it for active lanes), (2) eliminate the `thread float dprime[MAX_AXIS]` private array, which spills to device memory and is costly — instead, **write d' to u_out during forward sweep** (in-place backward sub from u_out), saving a large private-array spill. (3) Use `fma` and precompute `1/(b - a*cprime[i-1])` into cprime storage so denom division becomes a multiply. Store `cprime` and `inv_denom` in threadgroup memory; reuse across the warp.

```metal
#include <metal_stdlib>
using namespace metal;

constant constexpr uint MAX_AXIS = 384u;

// Precompute cprime[i] for i in [1, N-2] and inv_denom[i] = 1/(b - a*cprime[i-1])
// for i in [1, N-2]. Only thread 0 of the threadgroup does the serial recurrence;
// other threads wait at the barrier.
inline void init_coeffs(threadgroup float *cprime,
                        threadgroup float *invden,
                        uint N, float mu, uint tlid)
{
    if (tlid == 0u && N >= 3u) {
        float a = -mu;
        float b = 1.0f + 2.0f * mu;
        float c = -mu;
        // i = 1
        float inv_b = 1.0f / b;
        invden[1] = inv_b;
        cprime[1] = c * inv_b;
        // i = 2 .. N-2
        for (uint i = 2u; i <= N - 2u; ++i) {
            float denom = b - a * cprime[i - 1u];
            float inv = 1.0f / denom;
            invden[i] = inv;
            cprime[i] = c * inv;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Thomas solve. Stores d' into u_out during forward sweep, then does
// in-place backward substitution from u_out. Avoids private-array spill.
inline void thomas_line(device const float *u_in,
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

    // i = 1: rhs += mu * bd_lo
    float dp = (u_in[base + stride] + mu * bd_lo) * invden[1];
    u_out[base + stride] = dp;

    // i = 2 .. N - 3
    for (uint i = 2u; i + 1u < N - 1u; ++i) {
        float r = u_in[base + i * stride];
        dp = (r - a * dp) * invden[i];
        u_out[base + i * stride] = dp;
    }

    // i = N - 2: rhs += mu * bd_hi
    uint iN = N - 2u;
    float r = u_in[base + iN * stride] + mu * bd_hi;
    dp = (r - a * dp) * invden[iN];
    u_out[base + iN * stride] = dp;  // this is x[N-2]

    // Backward sub: read d' from u_out, overwrite with x.
    float xnext = dp;
    for (uint ii = N - 3u; ii >= 1u; --ii) {
        float dpi = u_out[base + ii * stride];
        float xi  = dpi - cprime[ii] * xnext;
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
    init_coeffs(cprime, invden, NX, mu, tlid);

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
    thomas_line(u_in, u_out, cprime, invden, base, 1u, NX, mu);
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
    init_coeffs(cprime, invden, NY, mu, tlid);

    uint i = gid.x;
    uint k = gid.y;
    if (i >= NX || k >= NZ) return;

    uint plane = NX * NY;
    uint base  = k * plane + i;

    if (i == 0u || i == NX - 1u || k == 0u || k == NZ - 1u) {
        for (uint j = 0u; j < NY; ++j) u_out[base + j * NX] = u_in[base + j * NX];
        return;
    }
    thomas_line(u_in, u_out, cprime, invden, base, NX, NY, mu);
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
    init_coeffs(cprime, invden, NZ, mu, tlid);

    uint i = gid.x;
    uint j = gid.y;
    if (i >= NX || j >= NY) return;

    uint plane = NX * NY;
    uint base  = j * NX + i;

    if (i == 0u || i == NX - 1u || j == 0u || j == NY - 1u) {
        for (uint k = 0u; k < NZ; ++k) u_out[base + k * plane] = u_in[base + k * plane];
        return;
    }
    thomas_line(u_in, u_out, cprime, invden, base, plane, NZ, mu);
}
```