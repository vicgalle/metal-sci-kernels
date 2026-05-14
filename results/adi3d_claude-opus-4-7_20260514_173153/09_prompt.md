## Task: adi3d

3D Locally-One-Dimensional (LOD) ADI for the heat equation. One timestep solves three constant-coefficient tridiagonal systems sequentially along x, then y, then z:
  (I - mu * Dxx) v1      = u^n         (x-sweep)
  (I - mu * Dyy) v2      = v1          (y-sweep)
  (I - mu * Dzz) u^{n+1} = v2          (z-sweep)
where mu = dt/h^2 (host uses mu = 0.5; LOD-ADI is unconditionally stable, no CFL). Each per-line system has constant tridiagonal entries
  -mu * v_{i-1} + (1 + 2 mu) * v_i + -mu * v_{i+1} = rhs_i,  1 <= i <= N-2
with Dirichlet endpoints v_0 = rhs_0, v_{N-1} = rhs_{N-1} (the line's two boundary cells, untouched by the solve).

Cube-face Dirichlet: every cell with i in {0, NX-1} OR j in {0, NY-1} OR k in {0, NZ-1} (any cube face) MUST stay at its initial value across the entire timestep. The harness enforces this convention: per sweep, lines whose two OFF-axis indices both sit strictly interior on the cube get a Thomas solve along the active axis; lines that touch a cube face in their off-axis indices copy u_in -> u_out unchanged. The result is that all six cube faces are preserved through every sub-step.

Storage is row-major float32 of shape (NZ, NY, NX) with i the fast (x) axis, j the middle (y) axis, k the slow (z) axis. Linear index: idx = (k * NY + j) * NX + i. NX, NY, and NZ are independent positive integers and need not be equal. The host calls three separate kernels -- adi_x, adi_y, adi_z -- in that order, ping-ponging two device buffers, with all dispatches sharing one command buffer for accurate end-to-end GPU timing of the n_steps run.

## Required kernel signature(s)

```
kernel void adi_x(device const float *u_in   [[buffer(0)]],
                  device       float *u_out  [[buffer(1)]],
                  constant uint      &NX     [[buffer(2)]],
                  constant uint      &NY     [[buffer(3)]],
                  constant uint      &NZ     [[buffer(4)]],
                  constant float     &mu     [[buffer(5)]],
                  uint2 gid [[thread_position_in_grid]]);
kernel void adi_y(device const float *u_in   [[buffer(0)]],
                  device       float *u_out  [[buffer(1)]],
                  constant uint      &NX     [[buffer(2)]],
                  constant uint      &NY     [[buffer(3)]],
                  constant uint      &NZ     [[buffer(4)]],
                  constant float     &mu     [[buffer(5)]],
                  uint2 gid [[thread_position_in_grid]]);
kernel void adi_z(device const float *u_in   [[buffer(0)]],
                  device       float *u_out  [[buffer(1)]],
                  constant uint      &NX     [[buffer(2)]],
                  constant uint      &NY     [[buffer(3)]],
                  constant uint      &NZ     [[buffer(4)]],
                  constant float     &mu     [[buffer(5)]],
                  uint2 gid [[thread_position_in_grid]]);

Dispatch geometry (host-fixed; identical pattern across the three kernels, with the two off-axis indices on gid.x and gid.y):
  adi_x: threadsPerGrid = (NY, NZ, 1), TG = (32, 1, 1).
         gid.x = j (off-axis y), gid.y = k (off-axis z).
  adi_y: threadsPerGrid = (NX, NZ, 1), TG = (32, 1, 1).
         gid.x = i (off-axis x), gid.y = k (off-axis z).
  adi_z: threadsPerGrid = (NX, NY, 1), TG = (32, 1, 1).
         gid.x = i (off-axis x), gid.y = j (off-axis y).
Convention: one thread owns one full Thomas line along the active axis. Each thread MUST early-exit if its gid is past the corresponding axis length. Boundary lines (those whose off-axis indices touch a cube face) MUST copy u_in -> u_out cell-by-cell.

If you cap the threadgroup with [[max_total_threads_per_threadgroup(W)]], place the attribute on the kernel declaration line itself, and remember the host dispatches TG = (32, 1, 1); a cap below 32 will be rejected. Buffers 0 and 1 are read/write and ping-ponged across timesteps, so do NOT assume u_in and u_out alias fixed addresses. The host calls adi_x -> adi_y -> adi_z back-to-back per timestep, with the output of one sweep being the input of the next; n_steps total timesteps share one command buffer.
```

## Your previous attempt

```metal
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
```

Result of previous attempt:
            N64_20: correct, 4.60 ms, 27.4 GB/s (effective, 24 B/cell/step across 3 sweeps) (13.7% of 200 GB/s)
            N96_15: correct, 8.55 ms, 37.3 GB/s (effective, 24 B/cell/step across 3 sweeps) (18.6% of 200 GB/s)
           N128_10: correct, 14.32 ms, 35.1 GB/s (effective, 24 B/cell/step across 3 sweeps) (17.6% of 200 GB/s)
  score (gmean of fraction): 0.1648

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

constant constexpr uint MAX_AXIS = 256u;

inline void init_cprime(threadgroup float *cprime,
                        uint N, float mu, uint tlid)
{
    if (tlid == 0u && N >= 3u) {
        float a = -mu;
        float b = 1.0f + 2.0f * mu;
        float c = -mu;
        cprime[1] = c / b;
        for (uint i = 2u; i < N - 1u; ++i) {
            cprime[i] = c / (b - a * cprime[i - 1u]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Thomas solve storing d' in private (per-thread) memory to avoid
// redundant device-memory writes during forward elimination.
inline void thomas_line_private(device const float *u_in,
                                device       float *u_out,
                                threadgroup const float *cprime,
                                uint base, uint stride, uint N, float mu)
{
    float a = -mu;
    float b = 1.0f + 2.0f * mu;

    float bd_lo = u_in[base];
    float bd_hi = u_in[base + (N - 1u) * stride];

    u_out[base]                     = bd_lo;
    u_out[base + (N - 1u) * stride] = bd_hi;

    if (N < 3u) return;

    thread float dprime[MAX_AXIS];

    // i = 1
    float dp = (u_in[base + stride] + mu * bd_lo) / b;
    dprime[1] = dp;

    // i = 2 .. N - 3
    for (uint i = 2u; i + 1u < N - 1u; ++i) {
        float denom = b - a * cprime[i - 1u];
        dp = (u_in[base + i * stride] - a * dp) / denom;
        dprime[i] = dp;
    }

    // i = N - 2
    uint iN = N - 2u;
    float denom = b - a * cprime[iN - 1u];
    dp = ((u_in[base + iN * stride] + mu * bd_hi) - a * dp) / denom;
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
                  uint2 gid  [[thread_position_in_grid]],
                  uint  tlid [[thread_index_in_threadgroup]])
{
    threadgroup float cprime[MAX_AXIS];
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
    thomas_line_private(u_in, u_out, cprime, base, 1u, NX, mu);
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
    thomas_line_private(u_in, u_out, cprime, base, NX, NY, mu);
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
    thomas_line_private(u_in, u_out, cprime, base, plane, NZ, mu);
}
```

Incumbent result:
            N64_20: correct, 3.74 ms, 33.6 GB/s (effective, 24 B/cell/step across 3 sweeps) (16.8% of 200 GB/s)
            N96_15: correct, 6.00 ms, 53.1 GB/s (effective, 24 B/cell/step across 3 sweeps) (26.5% of 200 GB/s)
           N128_10: correct, 10.82 ms, 46.5 GB/s (effective, 24 B/cell/step across 3 sweeps) (23.3% of 200 GB/s)
  score (gmean of fraction): 0.2181

## History

- iter  1: compile=OK | correct=False | score=N/A
- iter  2: compile=OK | correct=True | score=0.2181024628561033
- iter  3: compile=OK | correct=True | score=0.1521525170829217
- iter  4: compile=OK | correct=True | score=0.21328436718795288
- iter  5: compile=OK | correct=True | score=0.19805160545259845
- iter  6: compile=OK | correct=True | score=0.20229771121431084
- iter  7: compile=OK | correct=True | score=0.10188703327948384
- iter  8: compile=OK | correct=True | score=0.16483326452287472

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
