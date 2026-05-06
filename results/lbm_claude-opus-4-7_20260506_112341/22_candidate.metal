#include <metal_stdlib>
using namespace metal;

#define TX 16
#define TY 16
#define SX (TX + 2)
#define SY (TY + 2)

[[max_total_threads_per_threadgroup(256)]]
kernel void lbm_step(device const float *f_in   [[buffer(0)]],
                     device       float *f_out  [[buffer(1)]],
                     constant uint        &NX   [[buffer(2)]],
                     constant uint        &NY   [[buffer(3)]],
                     constant float       &tau  [[buffer(4)]],
                     uint2 gid  [[thread_position_in_grid]],
                     uint2 lid  [[thread_position_in_threadgroup]],
                     uint2 tgid [[threadgroup_position_in_grid]]) {
    threadgroup float tile[9][SY][SX];

    uint N = NX * NY;
    uint lx = lid.x;
    uint ly = lid.y;

    uint base_i = tgid.x * TX;
    uint base_j = tgid.y * TY;

    // Cooperative load of (TX+2)x(TY+2) halo region for all 9 planes.
    // 256 threads load (18*18 = 324) cells; each thread loads ~2 cells.
    uint tid = ly * TX + lx;       // 0..255
    uint total = SX * SY;          // 324

    for (uint t = tid; t < total; t += TX * TY) {
        uint sx = t % SX;
        uint sy = t / SX;
        // Global coord with halo offset: (-1, -1) corner.
        int gi = int(base_i) + int(sx) - 1;
        int gj = int(base_j) + int(sy) - 1;
        // Periodic wrap.
        uint ui = uint((gi + int(NX)) % int(NX));
        uint uj = uint((gj + int(NY)) % int(NY));
        uint gidx = uj * NX + ui;
        tile[0][sy][sx] = f_in[0u * N + gidx];
        tile[1][sy][sx] = f_in[1u * N + gidx];
        tile[2][sy][sx] = f_in[2u * N + gidx];
        tile[3][sy][sx] = f_in[3u * N + gidx];
        tile[4][sy][sx] = f_in[4u * N + gidx];
        tile[5][sy][sx] = f_in[5u * N + gidx];
        tile[6][sy][sx] = f_in[6u * N + gidx];
        tile[7][sy][sx] = f_in[7u * N + gidx];
        tile[8][sy][sx] = f_in[8u * N + gidx];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint i = gid.x;
    uint j = gid.y;
    if (i >= NX || j >= NY) return;

    // Local indices in shared tile (with +1 halo offset).
    uint sx = lx + 1;
    uint sy = ly + 1;

    // Pull streaming from threadgroup tile.
    // CX = {0, 1, 0,-1, 0, 1,-1,-1, 1}
    // CY = {0, 0, 1, 0,-1, 1, 1,-1,-1}
    float f0 = tile[0][sy    ][sx    ];
    float f1 = tile[1][sy    ][sx - 1];
    float f2 = tile[2][sy - 1][sx    ];
    float f3 = tile[3][sy    ][sx + 1];
    float f4 = tile[4][sy + 1][sx    ];
    float f5 = tile[5][sy - 1][sx - 1];
    float f6 = tile[6][sy - 1][sx + 1];
    float f7 = tile[7][sy + 1][sx + 1];
    float f8 = tile[8][sy + 1][sx - 1];

    // Moments.
    float rho     = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    float inv_rho = 1.0f / rho;
    float jx = (f1 - f3) + (f5 + f8) - (f6 + f7);
    float jy = (f2 - f4) + (f5 + f6) - (f7 + f8);
    float ux = jx * inv_rho;
    float uy = jy * inv_rho;

    float usq     = ux * ux + uy * uy;
    float omega   = 1.0f / tau;
    float one_m_w = 1.0f - omega;

    const float W0 = 4.0f / 9.0f;
    const float WS = 1.0f / 9.0f;
    const float WD = 1.0f / 36.0f;

    float orW0 = omega * W0 * rho;
    float orWS = omega * WS * rho;
    float orWD = omega * WD * rho;

    float A = fma(-1.5f, usq, 1.0f);

    uint idx = j * NX + i;

    // k=0
    f_out[0u * N + idx] = fma(one_m_w, f0, orW0 * A);
    // k=1
    {
        float cu = ux;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        f_out[1u * N + idx] = fma(one_m_w, f1, orWS * t);
    }
    // k=2
    {
        float cu = uy;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        f_out[2u * N + idx] = fma(one_m_w, f2, orWS * t);
    }
    // k=3
    {
        float cu = -ux;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        f_out[3u * N + idx] = fma(one_m_w, f3, orWS * t);
    }
    // k=4
    {
        float cu = -uy;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        f_out[4u * N + idx] = fma(one_m_w, f4, orWS * t);
    }
    // k=5
    {
        float cu = ux + uy;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        f_out[5u * N + idx] = fma(one_m_w, f5, orWD * t);
    }
    // k=6
    {
        float cu = uy - ux;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        f_out[6u * N + idx] = fma(one_m_w, f6, orWD * t);
    }
    // k=7
    {
        float cu = -(ux + uy);
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        f_out[7u * N + idx] = fma(one_m_w, f7, orWD * t);
    }
    // k=8
    {
        float cu = ux - uy;
        float t  = A + cu * fma(4.5f, cu, 3.0f);
        f_out[8u * N + idx] = fma(one_m_w, f8, orWD * t);
    }
}