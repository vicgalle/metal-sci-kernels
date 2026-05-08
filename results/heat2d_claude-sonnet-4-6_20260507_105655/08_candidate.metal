#include <metal_stdlib>
using namespace metal;

#define TILE_W 32
#define TILE_H 8

kernel void heat_step(device const float *u_in  [[buffer(0)]],
                      device       float *u_out [[buffer(1)]],
                      constant uint      &NX    [[buffer(2)]],
                      constant uint      &NY    [[buffer(3)]],
                      constant float     &alpha [[buffer(4)]],
                      uint2 gid   [[thread_position_in_grid]],
                      uint2 tid   [[thread_position_in_threadgroup]])
    __attribute__((max_total_threads_per_threadgroup(TILE_W * TILE_H)))
{
    // Shared memory tile with 1-cell halo: (TILE_H+2) x (TILE_W+2)
    threadgroup float smem[(TILE_H + 2) * (TILE_W + 2)];
    const uint sw = TILE_W + 2;

    const uint i = gid.x;
    const uint j = gid.y;

    const uint lx = tid.x + 1;
    const uint ly = tid.y + 1;

    // Safe (clamped) global indices for this thread
    const uint si = min(i, NX - 1);
    const uint sj = min(j, NY - 1);

    // Load center into smem
    smem[ly * sw + lx] = u_in[sj * NX + si];

    // Left halo (tid.x == 0 loads the column to the left)
    if (tid.x == 0) {
        uint hx = (i > 0) ? (si - 1) : 0;
        smem[ly * sw + 0] = u_in[sj * NX + hx];
    }
    // Right halo
    if (tid.x == TILE_W - 1) {
        uint hx = min(si + 1, NX - 1);
        smem[ly * sw + (TILE_W + 1)] = u_in[sj * NX + hx];
    }
    // Top halo (j-1 direction; j is row, so j-1 = "above" in memory)
    if (tid.y == 0) {
        uint hy = (j > 0) ? (sj - 1) : 0;
        smem[0 * sw + lx] = u_in[hy * NX + si];
    }
    // Bottom halo
    if (tid.y == TILE_H - 1) {
        uint hy = min(sj + 1, NY - 1);
        smem[(TILE_H + 1) * sw + lx] = u_in[hy * NX + si];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Guard after barrier so all threads participate in the barrier
    if (i >= NX || j >= NY) return;

    float center = smem[ly * sw + lx];

    if (i == 0 || j == 0 || i == NX - 1 || j == NY - 1) {
        u_out[j * NX + i] = center;
        return;
    }

    float l = smem[ly       * sw + (lx - 1)];
    float r = smem[ly       * sw + (lx + 1)];
    float up = smem[(ly - 1) * sw + lx];
    float dn = smem[(ly + 1) * sw + lx];

    u_out[j * NX + i] = center + alpha * (l + r + up + dn - 4.0f * center);
}