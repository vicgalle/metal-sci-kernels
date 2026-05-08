#include <metal_stdlib>
using namespace metal;

#define TX 16
#define TY 16

[[max_total_threads_per_threadgroup(TX*TY)]]
kernel void heat_step(device const float *u_in  [[buffer(0)]],
                      device       float *u_out [[buffer(1)]],
                      constant uint      &NX    [[buffer(2)]],
                      constant uint      &NY    [[buffer(3)]],
                      constant float     &alpha [[buffer(4)]],
                      uint2 gid  [[thread_position_in_grid]],
                      uint2 lid  [[thread_position_in_threadgroup]],
                      uint2 tgid [[threadgroup_position_in_grid]])
{
    threadgroup float tile[TY + 2][TX + 2];

    int i  = (int)gid.x;
    int j  = (int)gid.y;
    int lx = (int)lid.x;
    int ly = (int)lid.y;

    int nx = (int)NX;
    int ny = (int)NY;

    int base_x = (int)tgid.x * TX;
    int base_y = (int)tgid.y * TY;

    // Cooperative load: each thread loads one or more cells into tile[ly+1..][lx+1..].
    // Tile covers global region [base_x-1 .. base_x+TX] x [base_y-1 .. base_y+TY].
    // We loop with stride TX/TY so a TX*TY threadgroup covers the (TX+2)*(TY+2) tile.
    for (int ty = ly; ty < TY + 2; ty += TY) {
        int gy = base_y + ty - 1;
        int cgy = clamp(gy, 0, ny - 1);
        for (int tx = lx; tx < TX + 2; tx += TX) {
            int gx = base_x + tx - 1;
            int cgx = clamp(gx, 0, nx - 1);
            tile[ty][tx] = u_in[cgy * nx + cgx];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (i >= nx || j >= ny) return;

    uint idx = (uint)j * (uint)nx + (uint)i;

    if (i == 0 || j == 0 || i == nx - 1 || j == ny - 1) {
        u_out[idx] = tile[ly + 1][lx + 1];
        return;
    }

    float c = tile[ly + 1][lx + 1];
    float l = tile[ly + 1][lx    ];
    float r = tile[ly + 1][lx + 2];
    float d = tile[ly    ][lx + 1];
    float u = tile[ly + 2][lx + 1];

    u_out[idx] = c + alpha * ((l + r) + (d + u) - 4.0f * c);
}