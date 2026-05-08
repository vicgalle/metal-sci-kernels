Looking at the compile error, the `[[max_total_threads_per_threadgroup(N)]]` attribute must be placed on the kernel function itself, not as a trailing attribute. Let me fix the placement and keep the threadgroup memory tiling optimization.

The key optimization: cooperative threadgroup loading of a halo tile into shared memory reduces redundant global reads. Each interior cell shares 4 neighbors with adjacent threads — with tiling, each value is loaded once and reused. The attribute placement fix moves the hint to the correct position on the function declaration.

```metal
#include <metal_stdlib>
using namespace metal;

#define TILE_W 32
#define TILE_H 8

kernel void heat_step(device const float *u_in  [[buffer(0)]],
                      device       float *u_out [[buffer(1)]],
                      constant uint      &NX    [[buffer(2)]],
                      constant uint      &NY    [[buffer(3)]],
                      constant float     &alpha [[buffer(4)]],
                      uint2 gid  [[thread_position_in_grid]],
                      uint2 tid  [[thread_position_in_threadgroup]],
                      uint2 tgid [[threadgroup_position_in_grid]],
                      uint2 tgsize [[threads_per_threadgroup]])
    __attribute__((max_total_threads_per_threadgroup(TILE_W * TILE_H)))
{
    // Shared tile with 1-cell halo on each side: (TILE_H+2) x (TILE_W+2)
    threadgroup float smem[(TILE_H + 2) * (TILE_W + 2)];
    const uint sw = TILE_W + 2;

    uint i = gid.x;
    uint j = gid.y;

    // Clamped coords for safe halo loads at grid edges
    uint ci = (i < NX) ? i : NX - 1;
    uint cj = (j < NY) ? j : NY - 1;

    uint lx = tid.x + 1;  // local x with halo offset
    uint ly = tid.y + 1;  // local y with halo offset

    // --- Cooperative tile load ---
    // Center
    smem[ly * sw + lx] = u_in[cj * NX + ci];

    // Left halo: first column of threadgroup
    if (tid.x == 0) {
        uint hi = (i > 0) ? i - 1 : 0;
        smem[ly * sw + 0] = u_in[cj * NX + hi];
    }
    // Right halo: last column of threadgroup
    if (tid.x == TILE_W - 1) {
        uint hi = min(i + 1, NX - 1);
        smem[ly * sw + (TILE_W + 1)] = u_in[cj * NX + hi];
    }
    // Top halo: first row of threadgroup
    if (tid.y == 0) {
        uint hj = (j > 0) ? j - 1 : 0;
        smem[0 * sw + lx] = u_in[hj * NX + ci];
    }
    // Bottom halo: last row of threadgroup
    if (tid.y == TILE_H - 1) {
        uint hj = min(j + 1, NY - 1);
        smem[(TILE_H + 1) * sw + lx] = u_in[hj * NX + ci];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (i >= NX || j >= NY) return;

    float center = smem[ly * sw + lx];

    // Dirichlet BC: boundary cells copy through
    if (i == 0 || j == 0 || i == NX - 1 || j == NY - 1) {
        u_out[j * NX + i] = center;
        return;
    }

    float l = smem[ly       * sw + (lx - 1)];
    float r = smem[ly       * sw + (lx + 1)];
    float t = smem[(ly - 1) * sw + lx      ];
    float b = smem[(ly + 1) * sw + lx      ];

    u_out[j * NX + i] = center + alpha * (l + r + t + b - 4.0f * center);
}
```