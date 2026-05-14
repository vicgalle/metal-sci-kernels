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

    for (; n + 8u <= N; n += 8u) {
        uint i0 = idx;
        uint i1 = i0 + stride;
        uint i2 = i1 + stride;
        uint i3 = i2 + stride;
        uint i4 = i3 + stride;
        uint i5 = i4 + stride;
        uint i6 = i5 + stride;
        uint i7 = i6 + stride;

        float v0 = u_in[i0];
        float v1 = u_in[i1];
        float v2 = u_in[i2];
        float v3 = u_in[i3];
        float v4 = u_in[i4];
        float v5 = u_in[i5];
        float v6 = u_in[i6];
        float v7 = u_in[i7];

        u_out[i0] = v0;
        u_out[i1] = v1;
        u_out[i2] = v2;
        u_out[i3] = v3;
        u_out[i4] = v4;
        u_out[i5] = v5;
        u_out[i6] = v6;
        u_out[i7] = v7;

        idx = i7 + stride;
    }

    for (; n < N; ++n) {
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
        d2 = (u_in[idx] + HALF * dp) * INV2; dp = d2; idx += stride;
        d3 = (u_in[idx] + HALF * dp) * INV3; dp = d3; idx += stride;
        d4 = (u_in[idx] + HALF * dp) * INV4; dp = d4; idx += stride;
        d5 = (u_in[idx] + HALF * dp) * INV5; dp = d5; idx += stride;
        d6 = (u_in[idx] + HALF * dp) * INV6; dp = d6; idx += stride;
        d7 = (u_in[idx] + HALF * dp) * INV7; dp = d7; idx += stride;
        d8 = (u_in[idx] + HALF * dp) * INV8; dp = d8; idx += stride;

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
        if (limit > 2u) { d2 = (u_in[idx] + HALF * dp) * INV2; dp = d2; idx += stride; }
        if (limit > 3u) { d3 = (u_in[idx] + HALF * dp) * INV3; dp = d3; idx += stride; }
        if (limit > 4u) { d4 = (u_in[idx] + HALF * dp) * INV4; dp = d4; idx += stride; }
        if (limit > 5u) { d5 = (u_in[idx] + HALF * dp) * INV5; dp = d5; idx += stride; }
        if (limit > 6u) { d6 = (u_in[idx] + HALF * dp) * INV6; dp = d6; idx += stride; }
        if (limit > 7u) { d7 = (u_in[idx] + HALF * dp) * INV7; dp = d7; idx += stride; }

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

    if (ii >= 8u) { float xi = d8 - CP8 * xnext; u_out[bidx] = xi; xnext = xi; bidx -= stride; }
    if (ii >= 7u) { float xi = d7 - CP7 * xnext; u_out[bidx] = xi; xnext = xi; bidx -= stride; }
    if (ii >= 6u) { float xi = d6 - CP6 * xnext; u_out[bidx] = xi; xnext = xi; bidx -= stride; }
    if (ii >= 5u) { float xi = d5 - CP5 * xnext; u_out[bidx] = xi; xnext = xi; bidx -= stride; }
    if (ii >= 4u) { float xi = d4 - CP4 * xnext; u_out[bidx] = xi; xnext = xi; bidx -= stride; }
    if (ii >= 3u) { float xi = d3 - CP3 * xnext; u_out[bidx] = xi; xnext = xi; bidx -= stride; }
    if (ii >= 2u) { float xi = d2 - CP2 * xnext; u_out[bidx] = xi; xnext = xi; bidx -= stride; }
    if (ii >= 1u) { float xi = d1 - CP1 * xnext; u_out[bidx] = xi; }
}

#define DEFINE_THOMAS_FIXED_TAIL16(FNAME, NN, LIM)                                      \
inline void FNAME(device const float *u_in,                                             \
                  device       float *u_out,                                            \
                  uint base, uint stride)                                               \
{                                                                                       \
    const uint limit = (LIM);                                                           \
    const uint tail_start = (LIM) - 16u;                                                \
                                                                                        \
    uint last = base + ((NN) - 1u) * stride;                                            \
    float bd_lo = u_in[base];                                                           \
    float bd_hi = u_in[last];                                                           \
    u_out[base] = bd_lo;                                                                \
    u_out[last] = bd_hi;                                                                \
                                                                                        \
    uint idx = base + stride;                                                           \
                                                                                        \
    float d1 = (u_in[idx] + HALF * bd_lo) * INV1;                                       \
    float dp = d1; idx += stride;                                                       \
    float d2 = (u_in[idx] + HALF * dp) * INV2; dp = d2; idx += stride;                  \
    float d3 = (u_in[idx] + HALF * dp) * INV3; dp = d3; idx += stride;                  \
    float d4 = (u_in[idx] + HALF * dp) * INV4; dp = d4; idx += stride;                  \
    float d5 = (u_in[idx] + HALF * dp) * INV5; dp = d5; idx += stride;                  \
    float d6 = (u_in[idx] + HALF * dp) * INV6; dp = d6; idx += stride;                  \
    float d7 = (u_in[idx] + HALF * dp) * INV7; dp = d7; idx += stride;                  \
    float d8 = (u_in[idx] + HALF * dp) * INV8; dp = d8; idx += stride;                  \
                                                                                        \
    uint i = 9u;                                                                        \
    for (; i + 4u <= tail_start; i += 4u) {                                             \
        uint i0 = idx;                                                                  \
        uint i1 = i0 + stride;                                                          \
        uint i2 = i1 + stride;                                                          \
        uint i3 = i2 + stride;                                                          \
        float r0 = u_in[i0];                                                            \
        float r1 = u_in[i1];                                                            \
        float r2 = u_in[i2];                                                            \
        float r3 = u_in[i3];                                                            \
        float q0 = (r0 + HALF * dp) * INVINF;                                           \
        float q1 = (r1 + HALF * q0) * INVINF;                                           \
        float q2 = (r2 + HALF * q1) * INVINF;                                           \
        float q3 = (r3 + HALF * q2) * INVINF;                                           \
        u_out[i0] = q0;                                                                 \
        u_out[i1] = q1;                                                                 \
        u_out[i2] = q2;                                                                 \
        u_out[i3] = q3;                                                                 \
        dp = q3;                                                                        \
        idx = i3 + stride;                                                              \
    }                                                                                   \
    for (; i < tail_start; ++i) {                                                       \
        dp = (u_in[idx] + HALF * dp) * INVINF;                                          \
        u_out[idx] = dp;                                                                \
        idx += stride;                                                                  \
    }                                                                                   \
                                                                                        \
    float t0  = (u_in[idx] + HALF * dp) * INVINF; dp = t0;  idx += stride;              \
    float t1  = (u_in[idx] + HALF * dp) * INVINF; dp = t1;  idx += stride;              \
    float t2  = (u_in[idx] + HALF * dp) * INVINF; dp = t2;  idx += stride;              \
    float t3  = (u_in[idx] + HALF * dp) * INVINF; dp = t3;  idx += stride;              \
    float t4  = (u_in[idx] + HALF * dp) * INVINF; dp = t4;  idx += stride;              \
    float t5  = (u_in[idx] + HALF * dp) * INVINF; dp = t5;  idx += stride;              \
    float t6  = (u_in[idx] + HALF * dp) * INVINF; dp = t6;  idx += stride;              \
    float t7  = (u_in[idx] + HALF * dp) * INVINF; dp = t7;  idx += stride;              \
    float t8  = (u_in[idx] + HALF * dp) * INVINF; dp = t8;  idx += stride;              \
    float t9  = (u_in[idx] + HALF * dp) * INVINF; dp = t9;  idx += stride;              \
    float t10 = (u_in[idx] + HALF * dp) * INVINF; dp = t10; idx += stride;              \
    float t11 = (u_in[idx] + HALF * dp) * INVINF; dp = t11; idx += stride;              \
    float t12 = (u_in[idx] + HALF * dp) * INVINF; dp = t12; idx += stride;              \
    float t13 = (u_in[idx] + HALF * dp) * INVINF; dp = t13; idx += stride;              \
    float t14 = (u_in[idx] + HALF * dp) * INVINF; dp = t14; idx += stride;              \
    float t15 = (u_in[idx] + HALF * dp) * INVINF; dp = t15; idx += stride;              \
                                                                                        \
    dp = (u_in[idx] + HALF * dp + HALF * bd_hi) * INVINF;                               \
                                                                                        \
    float xnext = dp;                                                                   \
    u_out[idx] = xnext;                                                                 \
    uint bidx = idx - stride;                                                           \
                                                                                        \
    float xi;                                                                           \
    xi = t15 - CPINF * xnext; u_out[bidx] = xi; xnext = xi; bidx -= stride;             \
    xi = t14 - CPINF * xnext; u_out[bidx] = xi; xnext = xi; bidx -= stride;             \
    xi = t13 - CPINF * xnext; u_out[bidx] = xi; xnext = xi; bidx -= stride;             \
    xi = t12 - CPINF * xnext; u_out[bidx] = xi; xnext = xi; bidx -= stride;             \
    xi = t11 - CPINF * xnext; u_out[bidx] = xi; xnext = xi; bidx -= stride;             \
    xi = t10 - CPINF * xnext; u_out[bidx] = xi; xnext = xi; bidx -= stride;             \
    xi = t9  - CPINF * xnext; u_out[bidx] = xi; xnext = xi; bidx -= stride;             \
    xi = t8  - CPINF * xnext; u_out[bidx] = xi; xnext = xi; bidx -= stride;             \
    xi = t7  - CPINF * xnext; u_out[bidx] = xi; xnext = xi; bidx -= stride;             \
    xi = t6  - CPINF * xnext; u_out[bidx] = xi; xnext = xi; bidx -= stride;             \
    xi = t5  - CPINF * xnext; u_out[bidx] = xi; xnext = xi; bidx -= stride;             \
    xi = t4  - CPINF * xnext; u_out[bidx] = xi; xnext = xi; bidx -= stride;             \
    xi = t3  - CPINF * xnext; u_out[bidx] = xi; xnext = xi; bidx -= stride;             \
    xi = t2  - CPINF * xnext; u_out[bidx] = xi; xnext = xi; bidx -= stride;             \
    xi = t1  - CPINF * xnext; u_out[bidx] = xi; xnext = xi; bidx -= stride;             \
    xi = t0  - CPINF * xnext; u_out[bidx] = xi; xnext = xi; bidx -= stride;             \
                                                                                        \
    uint ii = tail_start - 1u;                                                          \
    while (ii >= 12u) {                                                                 \
        uint i0 = bidx;                                                                 \
        uint i1 = i0 - stride;                                                          \
        uint i2 = i1 - stride;                                                          \
        uint i3 = i2 - stride;                                                          \
        float v0 = u_out[i0];                                                           \
        float v1 = u_out[i1];                                                           \
        float v2 = u_out[i2];                                                           \
        float v3 = u_out[i3];                                                           \
        float x0 = v0 - CPINF * xnext;                                                  \
        float x1 = v1 - CPINF * x0;                                                     \
        float x2 = v2 - CPINF * x1;                                                     \
        float x3 = v3 - CPINF * x2;                                                     \
        u_out[i0] = x0;                                                                 \
        u_out[i1] = x1;                                                                 \
        u_out[i2] = x2;                                                                 \
        u_out[i3] = x3;                                                                 \
        xnext = x3;                                                                     \
        bidx = i3 - stride;                                                             \
        ii -= 4u;                                                                       \
    }                                                                                   \
    while (ii >= 9u) {                                                                  \
        xi = u_out[bidx] - CPINF * xnext;                                               \
        u_out[bidx] = xi;                                                               \
        xnext = xi;                                                                     \
        --ii;                                                                           \
        bidx -= stride;                                                                 \
    }                                                                                   \
                                                                                        \
    xi = d8 - CP8 * xnext; u_out[bidx] = xi; xnext = xi; bidx -= stride;                \
    xi = d7 - CP7 * xnext; u_out[bidx] = xi; xnext = xi; bidx -= stride;                \
    xi = d6 - CP6 * xnext; u_out[bidx] = xi; xnext = xi; bidx -= stride;                \
    xi = d5 - CP5 * xnext; u_out[bidx] = xi; xnext = xi; bidx -= stride;                \
    xi = d4 - CP4 * xnext; u_out[bidx] = xi; xnext = xi; bidx -= stride;                \
    xi = d3 - CP3 * xnext; u_out[bidx] = xi; xnext = xi; bidx -= stride;                \
    xi = d2 - CP2 * xnext; u_out[bidx] = xi; xnext = xi; bidx -= stride;                \
    xi = d1 - CP1 * xnext; u_out[bidx] = xi;                                            \
}

DEFINE_THOMAS_FIXED_TAIL16(thomas_line_mu05_64,  64u,  62u)
DEFINE_THOMAS_FIXED_TAIL16(thomas_line_mu05_96,  96u,  94u)
DEFINE_THOMAS_FIXED_TAIL16(thomas_line_mu05_128, 128u, 126u)

inline void thomas_dispatch_mu05(device const float *u_in,
                                 device       float *u_out,
                                 uint base, uint stride, uint N)
{
    if (N == 64u) {
        thomas_line_mu05_64(u_in, u_out, base, stride);
    } else if (N == 96u) {
        thomas_line_mu05_96(u_in, u_out, base, stride);
    } else if (N == 128u) {
        thomas_line_mu05_128(u_in, u_out, base, stride);
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

    thomas_dispatch_mu05(u_in, u_out, base, 1u, NX);
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

    thomas_dispatch_mu05(u_in, u_out, base, NX, NY);
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

    thomas_dispatch_mu05(u_in, u_out, base, plane, NZ);
}
```

Result of previous attempt:
  COMPILE FAILED: Error Domain=MTLLibraryErrorDomain Code=3 "program_source:358:1: warning: unused variable 'limit' [-Wunused-variable]
DEFINE_THOMAS_FIXED_TAIL16(thomas_line_mu05_64,  64u,  62u)
^
program_source:229:16: note: expanded from macro 'DEFINE_THOMAS_FIXED_TAIL16'
    const uint limit = (LIM);                                                           \
               ^
program_source:359:1: warning: unused variable 'limit' [-Wunused-variable]
DEFINE_THOMAS_FIXED_TAIL16(thomas_line_mu05_96,  96u,  94u)
^
program_source:229:16: note: expanded from macro 'DEFINE_THOMAS_FIXED_TAIL16'
    const uint limit = (LIM);                                                           \
               ^
program_source:360:1: warning: unused variable 'limit' [-Wunused-variable]
DEFINE_THOMAS_FIXED_TAIL16(thomas_line_mu05_128, 128u, 126u)
^
program_source:229:16: note: expanded from macro 'DEFINE_THOMAS_FIXED_TAIL16'
    const uint limit = (LIM);                                                           \
               ^
program_source:383:60: error: 'max_total_threads_per_threadgroup' attribute cannot be applied to types
                  uint2 gid [[thread_position_in_grid]]) [[max_total_threads_per_threadgroup(32)]]
                                                           ^
program_source:406:60: error: 'max_total_threads_per_threadgroup' attribute cannot be applied to types
                  uint2 gid [[thread_position_in_grid]]) [[max_total_threads_per_threadgroup(32)]]
                                                           ^
program_source:429:60: error: 'max_total_threads_per_threadgroup' attribute cannot be applied to types
                  uint2 gid [[thread_position_in_grid]]) [[max_total_threads_per_threadgroup(32)]]
                                                           ^
" UserInfo={NSLocalizedDescription=program_source:358:1: warning: unused variable 'limit' [-Wunused-variable]
DEFINE_THOMAS_FIXED_TAIL16(thomas_line_mu05_64,  64u,  62u)
^
program_source:229:16: note: expanded from macro 'DEFINE_THOMAS_FIXED_TAIL16'
    const uint limit = (LIM);                                                           \
               ^
program_source:359:1: warning: unused variable 'limit' [-Wunused-variable]
DEFINE_THOMAS_FIXED_TAIL16(thomas_line_mu05_96,  96u,  94u)
^
program_source:229:16: note: expanded from macro 'DEFINE_THOMAS_FIXED_TAIL16'
    const uint limit = (LIM);                                                           \
               ^
program_source:360:1: warning: unused variable 'limit' [-Wunused-variable]
DEFINE_THOMAS_FIXED_TAIL16(thomas_line_mu05_128, 128u, 126u)
^
program_source:229:16: note: expanded from macro 'DEFINE_THOMAS_FIXED_TAIL16'
    const uint limit = (LIM);                                                           \
               ^
program_source:383:60: error: 'max_total_threads_per_threadgroup' attribute cannot be applied to types
                  uint2 gid [[thread_position_in_grid]]) [[max_total_threads_per_threadgroup(32)]]
                                                           ^
program_source:406:60: error: 'max_total_threads_per_threadgroup' attribute cannot be applied to types
                  uint2 gid [[thread_position_in_grid]]) [[max_total_threads_per_threadgroup(32)]]
                                                           ^
program_source:429:60: error: 'max_total_threads_per_threadgroup' attribute cannot be applied to types
                  uint2 gid [[thread_position_in_grid]]) [[max_total_threads_per_threadgroup(32)]]
                                                           ^
}

## Current best (incumbent)

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

Incumbent result:
            N64_20: correct, 2.42 ms, 52.0 GB/s (effective, 24 B/cell/step across 3 sweeps) (26.0% of 200 GB/s)
            N96_15: correct, 3.47 ms, 91.7 GB/s (effective, 24 B/cell/step across 3 sweeps) (45.8% of 200 GB/s)
           N128_10: correct, 5.00 ms, 100.7 GB/s (effective, 24 B/cell/step across 3 sweeps) (50.3% of 200 GB/s)
  score (gmean of fraction): 0.3916

## History

- iter  1: compile=OK | correct=True | score=0.14679033652071152
- iter  2: compile=OK | correct=True | score=0.18269411660121826
- iter  3: compile=FAIL | correct=False | score=N/A
- iter  4: compile=OK | correct=True | score=0.17472259910848986
- iter  5: compile=OK | correct=True | score=0.3915793245743628
- iter  6: compile=FAIL | correct=False | score=N/A
- iter  7: compile=OK | correct=True | score=0.3869689771826847
- iter  8: compile=FAIL | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
