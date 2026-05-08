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
                      uint2 gid  [[thread_position_in_grid]],
                      uint2 tid  [[thread_position_in_threadgroup]],
                      uint2 tgid [[threadgroup_position_in_grid]])
    __attribute__((max_total_threads_per_threadgroup(TILE_W * TILE_H)))
{
    threadgroup float smem[SH * SW];

    // Base global coords for this threadgroup tile
    const int base_i = (int)(tgid.x * TILE_W);
    const int base_j = (int)(tgid.y * TILE_H);

    // Each thread's local position in smem (1-cell border offset)
    const uint lx = tid.x + 1;
    const uint ly = tid.y + 1;

    // Helper lambda: safe load with clamp
    // Load smem[sy * SW + sx] from u_in[gy * NX + gx], clamped
    auto safe_load = [&](uint sx, uint sy, int gx, int gy) {
        uint cx = (uint)clamp(gx, 0, (int)NX - 1);
        uint cy = (uint)clamp(gy, 0, (int)NY - 1);
        smem[sy * SW + sx] = u_in[cy * NX + cx];
    };

    // Every thread loads its center cell into smem
    safe_load(lx, ly, base_i + (int)tid.x, base_j + (int)tid.y);

    // Left halo: threads in column 0 load column base_i-1
    if (tid.x == 0) {
        safe_load(0, ly, base_i - 1, base_j + (int)tid.y);
    }
    // Right halo: threads in last column load column base_i+TILE_W
    if (tid.x == TILE_W - 1) {
        safe_load(SW - 1, ly, base_i + TILE_W, base_j + (int)tid.y);
    }
    // Top halo: threads in row 0 load row base_j-1
    if (tid.y == 0) {
        safe_load(lx, 0, base_i + (int)tid.x, base_j - 1);
    }
    // Bottom halo: threads in last row load row base_j+TILE_H
    if (tid.y == TILE_H - 1) {
        safe_load(lx, SH - 1, base_i + (int)tid.x, base_j + TILE_H);
    }

    // Corner halos: only 4 threads need to do this
    if (tid.x == 0 && tid.y == 0) {
        safe_load(0, 0, base_i - 1, base_j - 1);
    }
    if (tid.x == TILE_W - 1 && tid.y == 0) {
        safe_load(SW - 1, 0, base_i + TILE_W, base_j - 1);
    }
    if (tid.x == 0 && tid.y == TILE_H - 1) {
        safe_load(0, SH - 1, base_i - 1, base_j + TILE_H);
    }
    if (tid.x == TILE_W - 1 && tid.y == TILE_H - 1) {
        safe_load(SW - 1, SH - 1, base_i + TILE_W, base_j + TILE_H);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Global indices for this thread
    const uint i = gid.x;
    const uint j = gid.y;

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