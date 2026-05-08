#include <metal_stdlib>
using namespace metal;

#define TILE 16

kernel void __attribute__((max_total_threads_per_threadgroup(TILE*TILE)))
heat_step(device const float *u_in  [[buffer(0)]],
          device       float *u_out [[buffer(1)]],
          constant uint      &NX    [[buffer(2)]],
          constant uint      &NY    [[buffer(3)]],
          constant float     &alpha [[buffer(4)]],
          uint2 gid  [[thread_position_in_grid]],
          uint2 lid  [[thread_position_in_threadgroup]],
          uint2 tgid [[threadgroup_position_in_grid]])
{
    threadgroup float tile[TILE + 2][TILE + 2];

    uint i = gid.x;
    uint j = gid.y;

    int li = int(lid.x) + 1;
    int lj = int(lid.y) + 1;

    // Origin of this tile in global coords.
    int gi0 = int(tgid.x) * TILE;
    int gj0 = int(tgid.y) * TILE;

    // Center load (clamped so OOB threads still get a sane value; we won't
    // write OOB).
    uint ci = min(i, NX - 1);
    uint cj = min(j, NY - 1);
    tile[lj][li] = u_in[cj * NX + ci];

    // Halo loads. Each edge of the threadgroup loads its corresponding
    // halo strip; corners are handled by the corner threads.
    if (lid.x == 0) {
        int gi = gi0 - 1;
        uint si = uint(max(gi, 0));
        uint sj = min(uint(gj0 + int(lid.y)), NY - 1);
        tile[lj][0] = u_in[sj * NX + si];
    }
    if (lid.x == TILE - 1) {
        int gi = gi0 + TILE;
        uint si = uint(min(gi, int(NX) - 1));
        uint sj = min(uint(gj0 + int(lid.y)), NY - 1);
        tile[lj][TILE + 1] = u_in[sj * NX + si];
    }
    if (lid.y == 0) {
        int gj = gj0 - 1;
        uint sj = uint(max(gj, 0));
        uint si = min(uint(gi0 + int(lid.x)), NX - 1);
        tile[0][li] = u_in[sj * NX + si];
    }
    if (lid.y == TILE - 1) {
        int gj = gj0 + TILE;
        uint sj = uint(min(gj, int(NY) - 1));
        uint si = min(uint(gi0 + int(lid.x)), NX - 1);
        tile[TILE + 1][li] = u_in[sj * NX + si];
    }
    // Corners
    if (lid.x == 0 && lid.y == 0) {
        int gi = gi0 - 1, gj = gj0 - 1;
        uint si = uint(max(gi, 0));
        uint sj = uint(max(gj, 0));
        tile[0][0] = u_in[sj * NX + si];
    }
    if (lid.x == TILE - 1 && lid.y == 0) {
        int gi = gi0 + TILE, gj = gj0 - 1;
        uint si = uint(min(gi, int(NX) - 1));
        uint sj = uint(max(gj, 0));
        tile[0][TILE + 1] = u_in[sj * NX + si];
    }
    if (lid.x == 0 && lid.y == TILE - 1) {
        int gi = gi0 - 1, gj = gj0 + TILE;
        uint si = uint(max(gi, 0));
        uint sj = uint(min(gj, int(NY) - 1));
        tile[TILE + 1][0] = u_in[sj * NX + si];
    }
    if (lid.x == TILE - 1 && lid.y == TILE - 1) {
        int gi = gi0 + TILE, gj = gj0 + TILE;
        uint si = uint(min(gi, int(NX) - 1));
        uint sj = uint(min(gj, int(NY) - 1));
        tile[TILE + 1][TILE + 1] = u_in[sj * NX + si];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (i >= NX || j >= NY) return;

    uint idx = j * NX + i;
    float c = tile[lj][li];

    if (i == 0 || j == 0 || i == NX - 1 || j == NY - 1) {
        u_out[idx] = c;
        return;
    }

    float l = tile[lj][li - 1];
    float r = tile[lj][li + 1];
    float d = tile[lj - 1][li];
    float uu = tile[lj + 1][li];

    u_out[idx] = c + alpha * ((l + r) + (d + uu) - 4.0f * c);
}