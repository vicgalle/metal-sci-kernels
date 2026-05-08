#include <metal_stdlib>
using namespace metal;

#define TILE 16
#define HALO 1
#define SMEM (TILE + 2*HALO)

kernel void heat_step(
    device const float *u_in  [[buffer(0)]],
    device       float *u_out [[buffer(1)]],
    constant uint      &NX    [[buffer(2)]],
    constant uint      &NY    [[buffer(3)]],
    constant float     &alpha [[buffer(4)]],
    uint2 gid  [[thread_position_in_grid]],
    uint2 lid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    threadgroup float s[SMEM][SMEM];

    uint i = gid.x;
    uint j = gid.y;

    int li = int(lid.x);
    int lj = int(lid.y);

    int base_i = int(tgid.x) * TILE;
    int base_j = int(tgid.y) * TILE;

    int nx = int(NX);
    int ny = int(NY);

    // Cooperative load: 256 threads load 324 elements (18x18) into shared mem.
    int tid = lj * TILE + li;
    int total = SMEM * SMEM; // 324
    int nthreads = TILE * TILE; // 256

    for (int k = tid; k < total; k += nthreads) {
        int sy = k / SMEM;
        int sx = k - sy * SMEM;
        int gx = base_i + sx - HALO;
        int gy = base_j + sy - HALO;
        gx = clamp(gx, 0, nx - 1);
        gy = clamp(gy, 0, ny - 1);
        s[sy][sx] = u_in[gy * nx + gx];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (i >= NX || j >= NY) return;

    uint idx = j * NX + i;

    if (i == 0 || j == 0 || i == NX - 1 || j == NY - 1) {
        u_out[idx] = u_in[idx];
        return;
    }

    int sx = li + HALO;
    int sy = lj + HALO;

    float c  = s[sy    ][sx    ];
    float l  = s[sy    ][sx - 1];
    float r  = s[sy    ][sx + 1];
    float dn = s[sy - 1][sx    ];
    float up = s[sy + 1][sx    ];

    u_out[idx] = c + alpha * ((l + r) + (dn + up) - 4.0f * c);
}