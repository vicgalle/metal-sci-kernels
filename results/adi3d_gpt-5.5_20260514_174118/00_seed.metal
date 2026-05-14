// Seed for 3D Locally-One-Dimensional (LOD) ADI on the heat equation.
//
// One timestep solves three constant-coefficient tridiagonal systems
// sequentially along x, y, then z:
//
//     (I - mu * Dxx) v1     = u^n           // x-sweep (tridiag along x)
//     (I - mu * Dyy) v2     = v1            // y-sweep (tridiag along y)
//     (I - mu * Dzz) u^{n+1} = v2            // z-sweep (tridiag along z)
//
// where mu = dt/h^2 (host uses mu = 0.5; LOD-ADI is unconditionally
// stable so there is no CFL constraint). Each sweep's per-line system has
// the constant tridiagonal coefficients (-mu, 1 + 2mu, -mu):
//
//     -mu * v_{i-1} + (1 + 2 mu) * v_i + -mu * v_{i+1} = rhs_i,   1 <= i <= N-2
//
// with Dirichlet endpoints v_0 = rhs_0, v_{N-1} = rhs_{N-1} (the line's
// own boundary cells, untouched).
//
// Cube-face Dirichlet: every cell on a face of the cube (i in {0, NX-1}
// OR j in {0, NY-1} OR k in {0, NZ-1}) stays at its initial value across
// the whole timestep. The seed enforces this by: (a) the Thomas solver
// treats the swept-axis endpoints as Dirichlet sources, and (b) per
// sweep, lines whose two OFF-axis indices touch the cube boundary copy
// u_in -> u_out unchanged.
//
// Buffer layout (identical across the three kernels):
//
//   buffer 0: const float* u_in    (NX*NY*NZ, row-major)
//   buffer 1: device float* u_out  (NX*NY*NZ, row-major)
//   buffer 2: const uint&  NX
//   buffer 3: const uint&  NY
//   buffer 4: const uint&  NZ
//   buffer 5: const float& mu
//
// Linear index: idx(i, j, k) = (k * NY + j) * NX + i, with i the fast
// (x) axis, j the middle (y) axis, k the slow (z) axis. Per timestep
// the host calls adi_x, adi_y, adi_z in sequence, ping-ponging the two
// device buffers; all dispatches share one command buffer for accurate
// end-to-end GPU timing of the n_steps run.
//
// Dispatch geometry (host-fixed):
//
//   adi_x: threadsPerGrid = (NY, NZ, 1),  threadsPerThreadgroup = (32, 1, 1)
//   adi_y: threadsPerGrid = (NX, NZ, 1),  threadsPerThreadgroup = (32, 1, 1)
//   adi_z: threadsPerGrid = (NX, NY, 1),  threadsPerThreadgroup = (32, 1, 1)
//
// Convention: one thread owns one full Thomas line along the active
// axis. gid.x and gid.y carry the two off-axis indices of the line.

#include <metal_stdlib>
using namespace metal;

// MAX_AXIS bounds the longest axis the seed handles. 256 covers every
// in-distribution and held-out size (held-out NX = 256 is the largest).
constant constexpr uint MAX_AXIS = 256u;

// Compute c'_i for i = 1 .. N-2 of the constant-coef Thomas with
// a = -mu, b = 1 + 2 mu, c = -mu.  Recurrence: c'_1 = c/b; for i >= 2,
// c'_i = c / (b - a * c'_{i-1}).
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

// One constant-coef Thomas solve along an arbitrarily-strided axis.
//   base   = starting linear index of the line in the (NZ,NY,NX) buffer
//   stride = step between consecutive cells along the active axis
//            (1 for x, NX for y, NX*NY for z)
//   N      = axis length
// The two endpoints at base + 0*stride and base + (N-1)*stride are
// treated as Dirichlet sources (read from u_in, copied to u_out
// unchanged). The forward sweep streams d'_i into u_out[base + i*stride]
// for i = 1..N-2; the backward sweep reads them back and overwrites in
// place with the final solution.
inline void thomas_line(device const float *u_in,
                        device       float *u_out,
                        threadgroup const float *cprime,
                        uint base, uint stride, uint N, float mu)
{
    float a = -mu;
    float b = 1.0f + 2.0f * mu;

    float bd_lo = u_in[base];
    float bd_hi = u_in[base + (N - 1u) * stride];

    u_out[base]                       = bd_lo;
    u_out[base + (N - 1u) * stride]   = bd_hi;

    if (N < 3u) return;

    // Forward elim.  i = 1: Dirichlet correction r_1 += mu * bd_lo.
    float dp = (u_in[base + stride] + mu * bd_lo) / b;
    u_out[base + stride] = dp;

    // i = 2 .. N - 3.
    for (uint i = 2u; i + 1u < N - 1u; ++i) {
        float denom = b - a * cprime[i - 1u];
        dp = (u_in[base + i * stride] - a * dp) / denom;
        u_out[base + i * stride] = dp;
    }

    // i = N - 2: Dirichlet correction r_{N-2} += mu * bd_hi.
    {
        uint i = N - 2u;
        float denom = b - a * cprime[i - 1u];
        dp = ((u_in[base + i * stride] + mu * bd_hi) - a * dp) / denom;
        u_out[base + i * stride] = dp;
    }

    // Backward sub: x_{N-2} = dp (just computed); x_i = d'_i - c'_i * x_{i+1}
    // for i = N-3 .. 1.  In-place over u_out.
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
    init_cprime(cprime, NX, mu, tlid);

    uint j = gid.x;
    uint k = gid.y;
    if (j >= NY || k >= NZ) return;

    uint plane = NX * NY;
    uint base  = k * plane + j * NX;

    // Off-axis cube boundary: copy through.
    if (j == 0u || j == NY - 1u || k == 0u || k == NZ - 1u) {
        for (uint i = 0u; i < NX; ++i) u_out[base + i] = u_in[base + i];
        return;
    }
    thomas_line(u_in, u_out, cprime, base, 1u, NX, mu);
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
    uint base  = k * plane + i;       // j = 0; stride = NX along j

    if (i == 0u || i == NX - 1u || k == 0u || k == NZ - 1u) {
        for (uint j = 0u; j < NY; ++j) u_out[base + j * NX] = u_in[base + j * NX];
        return;
    }
    thomas_line(u_in, u_out, cprime, base, NX, NY, mu);
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
    uint base  = j * NX + i;          // k = 0; stride = NX*NY along k

    if (i == 0u || i == NX - 1u || j == 0u || j == NY - 1u) {
        for (uint k = 0u; k < NZ; ++k) u_out[base + k * plane] = u_in[base + k * plane];
        return;
    }
    thomas_line(u_in, u_out, cprime, base, plane, NZ, mu);
}
