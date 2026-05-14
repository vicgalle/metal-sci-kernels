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

constant constexpr float HALF = 0.5f;

// Thomas coefficients for mu = 0.5:
// a = c = -0.5, b = 2.0.
// c'_i quickly converges to sqrt(3)-2.
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

    uint limit = N - 2u; // final interior index

    float dp = (u_in[idx] + HALF * bd_lo) * INV1;
    u_out[idx] = dp;
    idx += stride;

    if (limit >= 9u) {
        dp = (u_in[idx] + HALF * dp) * INV2;
        u_out[idx] = dp;
        idx += stride;

        dp = (u_in[idx] + HALF * dp) * INV3;
        u_out[idx] = dp;
        idx += stride;

        dp = (u_in[idx] + HALF * dp) * INV4;
        u_out[idx] = dp;
        idx += stride;

        dp = (u_in[idx] + HALF * dp) * INV5;
        u_out[idx] = dp;
        idx += stride;

        dp = (u_in[idx] + HALF * dp) * INV6;
        u_out[idx] = dp;
        idx += stride;

        dp = (u_in[idx] + HALF * dp) * INV7;
        u_out[idx] = dp;
        idx += stride;

        dp = (u_in[idx] + HALF * dp) * INV8;
        u_out[idx] = dp;
        idx += stride;

        for (uint i = 9u; i < limit; ++i) {
            dp = (u_in[idx] + HALF * dp) * INVINF;
            u_out[idx] = dp;
            idx += stride;
        }
    } else {
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
        if (limit > 7u) {
            dp = (u_in[idx] + HALF * dp) * INV7;
            u_out[idx] = dp;
            idx += stride;
        }
    }

    dp = (u_in[idx] + HALF * dp + HALF * bd_hi) * inv_for_index(limit);
    float xnext = dp;
    u_out[idx] = xnext;

    uint ii   = limit - 1u;
    uint bidx = idx - stride;

    while (ii >= 9u) {
        float xi = u_out[bidx] - CPINF * xnext;
        u_out[bidx] = xi;
        xnext = xi;
        --ii;
        bidx -= stride;
    }

    if (ii >= 8u) {
        float xi = u_out[bidx] - CP8 * xnext;
        u_out[bidx] = xi; xnext = xi; bidx -= stride;
    }
    if (ii >= 7u) {
        float xi = u_out[bidx] - CP7 * xnext;
        u_out[bidx] = xi; xnext = xi; bidx -= stride;
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
    uint j = gid.x;
    uint k = gid.y;
    if (j >= NY || k >= NZ) return;

    uint plane = NX * NY;
    uint base  = k * plane + j * NX;

    if (j == 0u || j == NY - 1u || k == 0u || k == NZ - 1u) {
        copy_line(u_in, u_out, base, 1u, NX);
        return;
    }

    thomas_line_mu05(u_in, u_out, base, 1u, NX);
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
        copy_line(u_in, u_out, base, NX, NY);
        return;
    }

    thomas_line_mu05(u_in, u_out, base, NX, NY);
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
        copy_line(u_in, u_out, base, plane, NZ);
        return;
    }

    thomas_line_mu05(u_in, u_out, base, plane, NZ);
}
```

Result of previous attempt:
            N64_20: correct, 4.20 ms, 30.0 GB/s (effective, 24 B/cell/step across 3 sweeps) (15.0% of 200 GB/s)
            N96_15: correct, 8.02 ms, 39.7 GB/s (effective, 24 B/cell/step across 3 sweeps) (19.8% of 200 GB/s)
           N128_10: correct, 12.28 ms, 41.0 GB/s (effective, 24 B/cell/step across 3 sweeps) (20.5% of 200 GB/s)
  score (gmean of fraction): 0.1827

## History

- iter  0: compile=OK | correct=True | score=0.17155591493393785
- iter  1: compile=OK | correct=True | score=0.14679033652071152
- iter  2: compile=OK | correct=True | score=0.18269411660121826

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
