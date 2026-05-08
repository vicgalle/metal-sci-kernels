#include <metal_stdlib>
using namespace metal;

#define TILE_W 32
#define TILE_H 8
#define SW (TILE_W + 2)
#define SH (TILE_H + 2)

kernel void heat_step(device const float *u_in  [[buffer(0)]],
                      device       float *u_out [[buffer(1)]],
                      constant uint      &NX    [[buffer(2)]],
                      constant uint      &NY    [[buffer(3)]],
                      constant float     &alpha [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]],
                      uint2 tid [[thread_position_in_threadgroup]],
                      uint2 tgid [[threadgroup_position_in_grid]])
    __attribute__((max_total_threads_per_threadgroup(TILE_W * TILE_H)))
{
    threadgroup float smem[SH * SW];

    // smem indices for this thread's center cell (with 1-cell border)
    const uint lx = tid.x + 1;
    const uint ly = tid.y + 1;

    // Global indices for this thread
    const uint i = gid.x;
    const uint j = gid.y;

    // Clamp to valid range for loading (guards against out-of-bounds reads)
    const uint ci = min(i, NX - 1);
    const uint cj = min(j, NY - 1);

    // Load center
    smem[ly * SW + lx] = u_in[cj * NX + ci];

    // Halo loads: use actual neighboring global coords, clamped to grid
    // Left halo column (tid.x == 0)
    if (tid.x == 0) {
        uint hx = (i > 0) ? (i - 1) : 0;
        hx = min(hx, NX - 1);
        smem[ly * SW + 0] = u_in[cj * NX + hx];
    }
    // Right halo column (tid.x == TILE_W - 1, but last tile may be smaller)
    // Use tid.x == TILE_W-1 OR actual last thread in threadgroup
    if (tid.x == TILE_W - 1 || i == NX - 1) {
        uint hx = min(i + 1, NX - 1);
        smem[ly * SW + (lx + 1)] = u_in[cj * NX + hx];
    }
    // Top halo row (tid.y == 0)
    if (tid.y == 0) {
        uint hy = (j > 0) ? (j - 1) : 0;
        hy = min(hy, NY - 1);
        smem[0 * SW + lx] = u_in[hy * NX + ci];
    }
    // Bottom halo row
    if (tid.y == TILE_H - 1 || j == NY - 1) {
        uint hy = min(j + 1, NY - 1);
        smem[(ly + 1) * SW + lx] = u_in[hy * NX + ci];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Guard after barrier
    if (i >= NX || j >= NY) return;

    float center = smem[ly * SW + lx];

    if (i == 0 || j == 0 || i == NX - 1 || j == NY - 1) {
        u_out[j * NX + i] = center;
        return;
    }

    float left  = smem[ly       * SW + (lx - 1)];
    float right = smem[ly       * SW + (lx + 1)];
    float up    = smem[(ly - 1) * SW + lx];
    float down  = smem[(ly + 1) * SW + lx];

    u_out[j * NX + i] = center + alpha * (left + right + up + down - 4.0f * center);
}