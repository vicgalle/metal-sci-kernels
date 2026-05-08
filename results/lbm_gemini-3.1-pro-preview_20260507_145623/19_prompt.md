## Task: lbm

D2Q9 lattice Boltzmann method, fused pull-streaming + BGK collision, periodic boundary conditions. Distribution functions are stored SoA: f[k * NX*NY + j*NX + i] for k in [0, 9), j in [0, NY), i in [0, NX), float32 row-major.

Per timestep, per cell (i, j):
  1) PULL stream: f_streamed[k] = f_in[k, (i - CX[k]) mod NX,
                                          (j - CY[k]) mod NY]
  2) Moments: rho = sum_k f_streamed[k];
     u = (sum_k CX[k] * f_streamed[k]) / rho; v likewise.
  3) BGK collision: f_out[k] = f_streamed[k]
       - (1/tau) (f_streamed[k] - f_eq[k])
     with f_eq[k] = W[k] * rho *
       (1 + 3 (CX[k] u + CY[k] v)
          + 4.5 (CX[k] u + CY[k] v)^2 - 1.5 (u^2 + v^2)).
Velocity table CX[9] = {0, 1, 0,-1, 0, 1,-1,-1, 1};
                CY[9] = {0, 0, 1, 0,-1, 1, 1,-1,-1};
weights W[9] = {4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36}.

The host runs the kernel n_steps times with two buffers ping-ponged each call. Effective DRAM traffic per step is 72 bytes/cell (9 reads + 9 writes), so the roofline is BW-bound.

## Required kernel signature(s)

```
kernel void lbm_step(device const float *f_in   [[buffer(0)]],
                     device       float *f_out  [[buffer(1)]],
                     constant uint        &NX   [[buffer(2)]],
                     constant uint        &NY   [[buffer(3)]],
                     constant float       &tau  [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]]);

Grid is dispatched 2-D as `threadsPerGrid = (NX, NY)`, one thread per output cell — guard with `if (i >= NX || j >= NY) return;`. Each thread MUST update exactly one cell; the host will not shrink the dispatch if you process multiple cells per thread, so extra threads just idle. SoA layout MUST be preserved on buffers 0 and 1; the kernel may use any internal layout/optimization (threadgroup tiling, simdgroup ops, etc.).
```

## Your previous attempt

```metal
#include <metal_stdlib>
using namespace metal;

constexpr float CX[9] = {0.0f,  1.0f,  0.0f, -1.0f,  0.0f,  1.0f, -1.0f, -1.0f,  1.0f};
constexpr float CY[9] = {0.0f,  0.0f,  1.0f,  0.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f};

kernel void lbm_step(device const float *f_in   [[buffer(0)]],
                     device       float *f_out  [[buffer(1)]],
                     constant uint        &NX   [[buffer(2)]],
                     constant uint        &NY   [[buffer(3)]],
                     constant float       &tau  [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]])
                     [[max_total_threads_per_threadgroup(256)]]
{
    uint i = gid.x;
    uint j = gid.y;
    uint nx = NX;
    uint ny = NY;
    
    if (i >= nx || j >= ny) return;

    int inx = int(nx);
    int iny = int(ny);
    int N = inx * iny;
    int idx = int(j) * inx + int(i);

    // Fast branchless periodic boundary offset calculation
    int im1 = select(-1, inx - 1, i == 0);
    int ip1 = select(1, 1 - inx, i == nx - 1);
    int jm1 = select(-inx, inx * (iny - 1), j == 0);
    int jp1 = select(inx, -inx * (iny - 1), j == ny - 1);

    const int off[9] = {
        0, im1, jm1, ip1, jp1, im1 + jm1, ip1 + jm1, ip1 + jp1, im1 + jp1
    };

    device const float *p_in = f_in + idx;
    device       float *p_out = f_out + idx;

    float f[9];
    float rho = 0.0f;
    float ux = 0.0f;
    float uy = 0.0f;

    // Pull streaming and compute raw moments
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        float v = p_in[k * N + off[k]];
        f[k] = v;
        rho += v;
        ux += CX[k] * v;
        uy += CY[k] * v;
    }

    // Normalize moments
    float inv_rho = 1.0f / rho;
    ux *= inv_rho;
    uy *= inv_rho;

    // Derived scalars for algebraic simplification of BGK
    float usq = fma(ux, ux, uy * uy);
    float inv_tau = 1.0f / tau;
    float oma = 1.0f - inv_tau;
    float rho_tau = rho * inv_tau;
    float term0 = fma(-1.5f, usq, 1.0f);

    constexpr float W0 = 4.0f / 9.0f;
    constexpr float W1 = 1.0f / 9.0f;
    constexpr float W5 = 1.0f / 36.0f;

    float w0_rho_tau = W0 * rho_tau;
    float w1_rho_tau = W1 * rho_tau;
    float w5_rho_tau = W5 * rho_tau;

    const float w_rho_tau[9] = {
        w0_rho_tau,
        w1_rho_tau, w1_rho_tau, w1_rho_tau, w1_rho_tau,
        w5_rho_tau, w5_rho_tau, w5_rho_tau, w5_rho_tau
    };

    // Collision execution explicitly resolving to chained FMAs
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        float cu = CX[k] * ux + CY[k] * uy;
        float poly = fma(cu, fma(cu, 4.5f, 3.0f), term0);
        float feq_tau = w_rho_tau[k] * poly;
        p_out[k * N] = fma(f[k], oma, feq_tau);
    }
}
```

Result of previous attempt:
  COMPILE FAILED: Error Domain=MTLLibraryErrorDomain Code=3 "program_source:4:17: error: program scope variable must reside in constant address space
constexpr float CX[9] = {0.0f,  1.0f,  0.0f, -1.0f,  0.0f,  1.0f, -1.0f, -1.0f,  1.0f};
                ^
program_source:5:17: error: program scope variable must reside in constant address space
constexpr float CY[9] = {0.0f,  0.0f,  1.0f,  0.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f};
                ^
program_source:13:24: error: 'max_total_threads_per_threadgroup' attribute cannot be applied to types
                     [[max_total_threads_per_threadgroup(256)]]
                       ^
" UserInfo={NSLocalizedDescription=program_source:4:17: error: program scope variable must reside in constant address space
constexpr float CX[9] = {0.0f,  1.0f,  0.0f, -1.0f,  0.0f,  1.0f, -1.0f, -1.0f,  1.0f};
                ^
program_source:5:17: error: program scope variable must reside in constant address space
constexpr float CY[9] = {0.0f,  0.0f,  1.0f,  0.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f};
                ^
program_source:13:24: error: 'max_total_threads_per_threadgroup' attribute cannot be applied to types
                     [[max_total_threads_per_threadgroup(256)]]
                       ^
}

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

constant float CX[9] = {0.0f,  1.0f,  0.0f, -1.0f,  0.0f,  1.0f, -1.0f, -1.0f,  1.0f};
constant float CY[9] = {0.0f,  0.0f,  1.0f,  0.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f};
constant float W[9] = {
    4.0f / 9.0f,
    1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
    1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f
};

kernel void lbm_step(device const float *f_in   [[buffer(0)]],
                     device       float *f_out  [[buffer(1)]],
                     constant uint        &NX   [[buffer(2)]],
                     constant uint        &NY   [[buffer(3)]],
                     constant float       &tau  [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint j = gid.y;
    uint nx = NX;
    uint ny = NY;
    
    if (i >= nx || j >= ny) return;

    int inx = int(nx);
    int iny = int(ny);
    uint N = nx * ny;
    uint idx = j * nx + i;

    // Fast 1D periodic boundary offset calculation
    int im1 = (i == 0) ? (inx - 1) : -1;
    int ip1 = (i == nx - 1) ? (1 - inx) : 1;
    int jm1 = (j == 0) ? (inx * (iny - 1)) : -inx;
    int jp1 = (j == ny - 1) ? -(inx * (iny - 1)) : inx;

    const int off[9] = {
        0,
        im1,
        jm1,
        ip1,
        jp1,
        im1 + jm1,
        ip1 + jm1,
        ip1 + jp1,
        im1 + jp1
    };

    float f[9];
    float rho = 0.0f;
    float ux = 0.0f;
    float uy = 0.0f;

    // Pull streaming and moments computation
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        float v = f_in[k * N + idx + uint(off[k])];
        f[k] = v;
        rho += v;
        ux += CX[k] * v;
        uy += CY[k] * v;
    }

    float inv_rho = 1.0f / rho;
    ux *= inv_rho;
    uy *= inv_rho;

    float usq = ux * ux + uy * uy;
    float inv_tau = 1.0f / tau;

    // BGK collision
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        float cu = CX[k] * ux + CY[k] * uy;
        float feq = W[k] * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * usq);
        f_out[k * N + idx] = f[k] - inv_tau * (f[k] - feq);
    }
}
```

Incumbent result:
          64x64_50: correct, 0.23 ms, 64.4 GB/s (effective, 72 B/cell) (32.2% of 200 GB/s)
       128x128_100: correct, 1.15 ms, 102.4 GB/s (effective, 72 B/cell) (51.2% of 200 GB/s)
       256x256_100: correct, 2.30 ms, 205.1 GB/s (effective, 72 B/cell) (102.6% of 200 GB/s)
  score (gmean of fraction): 0.5531

## History

- iter 11: compile=OK | correct=True | score=0.3909012300348845
- iter 12: compile=OK | correct=True | score=0.3624193042474191
- iter 13: compile=OK | correct=True | score=0.5530764161168673
- iter 14: compile=OK | correct=True | score=0.48881450123883946
- iter 15: compile=OK | correct=True | score=0.36496197947723824
- iter 16: compile=FAIL | correct=False | score=N/A
- iter 17: compile=OK | correct=True | score=0.3566821793420694
- iter 18: compile=FAIL | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
