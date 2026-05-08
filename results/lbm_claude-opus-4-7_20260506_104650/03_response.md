**Optimization:** The kernel is BW-bound and small grids (64×64, 128×128) don't fill the GPU well — they're latency/launch-bound, not BW-bound. To improve those, I'll have each thread process **2 cells along x** (register blocking), halving the dispatched threadgroup count and amortizing index computation. Adjacent cells in x share the `row_j/jm1/jp1` row bases and most neighbor indices, so we save address arithmetic. I'll also dispatch with `[[max_total_threads_per_threadgroup(256)]]` hint and use `fma`-friendly arithmetic. The grid is dispatched as `(NX, NY)` per spec — to keep that contract, I gate the second cell on `(i+1 < NX)` and use a strided check; but since the host dispatches `(NX, NY)` exactly, every thread does work for cell `(i,j)` as before — I can instead process 2 distributions at once via vectorized load patterns. Sticking with single-cell-per-thread but reorganizing for better ILP and using `precise::divide` → `fast::divide` for `1/rho`.

```metal
#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(256)]]
kernel void lbm_step(device const float *f_in   [[buffer(0)]],
                     device       float *f_out  [[buffer(1)]],
                     constant uint        &NX   [[buffer(2)]],
                     constant uint        &NY   [[buffer(3)]],
                     constant float       &tau  [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint j = gid.y;
    if (i >= NX || j >= NY) return;

    uint N = NX * NY;

    uint im1 = select(i - 1u, NX - 1u, i == 0u);
    uint ip1 = select(i + 1u, 0u,      i == NX - 1u);
    uint jm1 = select(j - 1u, NY - 1u, j == 0u);
    uint jp1 = select(j + 1u, 0u,      j == NY - 1u);

    uint row_j   = j   * NX;
    uint row_jm1 = jm1 * NX;
    uint row_jp1 = jp1 * NX;

    uint idx  = row_j   + i;
    uint idxL = row_j   + im1;
    uint idxR = row_j   + ip1;
    uint idxD = row_jm1 + i;
    uint idxU = row_jp1 + i;
    uint idxDL = row_jm1 + im1;
    uint idxDR = row_jm1 + ip1;
    uint idxUL = row_jp1 + im1;
    uint idxUR = row_jp1 + ip1;

    // Pull-stream loads.
    float f0 = f_in[0u * N + idx ];
    float f1 = f_in[1u * N + idxL];
    float f2 = f_in[2u * N + idxD];
    float f3 = f_in[3u * N + idxR];
    float f4 = f_in[4u * N + idxU];
    float f5 = f_in[5u * N + idxDL];
    float f6 = f_in[6u * N + idxDR];
    float f7 = f_in[7u * N + idxUR];
    float f8 = f_in[8u * N + idxUL];

    // Moments.
    float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    float inv_rho = fast::divide(1.0f, rho);
    float mx = (f1 + f5 + f8) - (f3 + f6 + f7);
    float my = (f2 + f5 + f6) - (f4 + f7 + f8);
    float ux = mx * inv_rho;
    float uy = my * inv_rho;

    float usq = ux * ux + uy * uy;
    float inv_tau = 1.0f / tau;
    float omt = 1.0f - inv_tau;

    const float W0 = 4.0f / 9.0f;
    const float W1 = 1.0f / 9.0f;
    const float W5 = 1.0f / 36.0f;

    float rit = rho * inv_tau;
    float A0 = W0 * rit;
    float A1 = W1 * rit;
    float A5 = W5 * rit;

    float c1 = 1.0f - 1.5f * usq;

    // k=0
    float feq0 = A0 * c1;

    // x-axis pair (k=1, k=3): cu = ±ux
    float sx = c1 + 4.5f * ux * ux;
    float tx = 3.0f * ux;
    float feq1 = A1 * (sx + tx);
    float feq3 = A1 * (sx - tx);

    // y-axis pair (k=2, k=4): cu = ±uy
    float sy = c1 + 4.5f * uy * uy;
    float ty = 3.0f * uy;
    float feq2 = A1 * (sy + ty);
    float feq4 = A1 * (sy - ty);

    // diagonal (k=5, k=7): cu = ±(ux+uy)
    float d1 = ux + uy;
    float sd1 = c1 + 4.5f * d1 * d1;
    float td1 = 3.0f * d1;
    float feq5 = A5 * (sd1 + td1);
    float feq7 = A5 * (sd1 - td1);

    // anti-diagonal (k=6, k=8): cu = ±(uy-ux)
    float d2 = uy - ux;
    float sd2 = c1 + 4.5f * d2 * d2;
    float td2 = 3.0f * d2;
    float feq6 = A5 * (sd2 + td2);
    float feq8 = A5 * (sd2 - td2);

    f_out[0u * N + idx] = fma(omt, f0, feq0);
    f_out[1u * N + idx] = fma(omt, f1, feq1);
    f_out[2u * N + idx] = fma(omt, f2, feq2);
    f_out[3u * N + idx] = fma(omt, f3, feq3);
    f_out[4u * N + idx] = fma(omt, f4, feq4);
    f_out[5u * N + idx] = fma(omt, f5, feq5);
    f_out[6u * N + idx] = fma(omt, f6, feq6);
    f_out[7u * N + idx] = fma(omt, f7, feq7);
    f_out[8u * N + idx] = fma(omt, f8, feq8);
}
```