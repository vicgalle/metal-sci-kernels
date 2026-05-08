Looking at the failure: `[[max_total_threads_per_threadgroup(256)]]` must be placed on the kernel function itself, not as a trailing attribute after the closing `}`. Let me fix that and also fix the threadgroup tiling approach properly.

**Optimization:** Threadgroup memory tiling with a (TILE_W+2)×(TILE_H+2) shared buffer reduces global memory reads from 5 to ~1.25 per cell (neighbors are reused from smem). The `[[max_total_threads_per_threadgroup(256)]]` attribute is placed correctly on the function signature. This amortizes the 5-point stencil neighbor loads across the tile, which should help the 256×256 case that's currently bandwidth-limited.

```metal
#include <metal_stdlib>
using namespace metal;

constant uint TILE_W = 16;
constant uint TILE_H = 16;

kernel void heat_step(device const float *u_in  [[buffer(0)]],
                      device       float *u_out [[buffer(1)]],
                      constant uint      &NX    [[buffer(2)]],
                      constant uint      &NY    [[buffer(3)]],
                      constant float     &alpha [[buffer(4)]],
                      uint2 gid  [[thread_position_in_grid]],
                      uint2 tid  [[thread_position_in_threadgroup]],
                      uint2 tgid [[threadgroup_position_in_grid]])
    [[max_total_threads_per_threadgroup(256)]]
{
    // Shared tile with 1-cell halo on each side: (TILE_W+2) x (TILE_H+2)
    threadgroup float smem[(TILE_H + 2) * (TILE_W + 2)];
    const uint sw = TILE_W + 2;

    uint i = gid.x;
    uint j = gid.y;

    uint lx = tid.x + 1;  // local x with halo offset
    uint ly = tid.y + 1;  // local y with halo offset

    // Clamp coordinates for safe OOB loads (grid may be padded)
    uint ci = min(i, NX - 1);
    uint cj = min(j, NY - 1);

    // Load interior of tile
    smem[ly * sw + lx] = u_in[cj * NX + ci];

    // Left halo
    if (tid.x == 0) {
        uint hi = (i > 0) ? (i - 1) : 0;
        uint hci = min(hi, NX - 1);
        smem[ly * sw + 0] = u_in[cj * NX + hci];
    }
    // Right halo
    if (tid.x == TILE_W - 1) {
        uint hi = i + 1;
        uint hci = min(hi, NX - 1);
        smem[ly * sw + (TILE_W + 1)] = u_in[cj * NX + hci];
    }
    // Top halo (j-1)
    if (tid.y == 0) {
        uint hj = (j > 0) ? (j - 1) : 0;
        uint hcj = min(hj, NY - 1);
        smem[0 * sw + lx] = u_in[hcj * NX + ci];
    }
    // Bottom halo (j+1)
    if (tid.y == TILE_H - 1) {
        uint hj = j + 1;
        uint hcj = min(hj, NY - 1);
        smem[(TILE_H + 1) * sw + lx] = u_in[hcj * NX + ci];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (i >= NX || j >= NY) return;

    // Boundary cells: copy unchanged (Dirichlet BC)
    if (i == 0 || j == 0 || i == NX - 1 || j == NY - 1) {
        u_out[j * NX + i] = smem[ly * sw + lx];
        return;
    }

    float c = smem[ly       * sw + lx    ];
    float l = smem[ly       * sw + lx - 1];
    float r = smem[ly       * sw + lx + 1];
    float t = smem[(ly - 1) * sw + lx    ];
    float b = smem[(ly + 1) * sw + lx    ];

    u_out[j * NX + i] = c + alpha * (l + r + t + b - 4.0f * c);
}
```