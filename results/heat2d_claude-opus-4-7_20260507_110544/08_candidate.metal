#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(256)]]
kernel void heat_step(device const float *u_in  [[buffer(0)]],
                      device       float *u_out [[buffer(1)]],
                      constant uint      &NX    [[buffer(2)]],
                      constant uint      &NY    [[buffer(3)]],
                      constant float     &alpha [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]])
{
    uint i = gid.x;
    uint j = gid.y;
    if (i >= NX || j >= NY) return;

    uint nx = NX;
    uint idx = j * nx + i;

    if (i == 0 || j == 0 || i == nx - 1 || j == NY - 1) {
        u_out[idx] = u_in[idx];
        return;
    }

    device const float *row  = u_in + idx;
    float c = row[0];
    float l = row[-1];
    float r = row[1];
    float d = row[-(int)nx];
    float up = row[nx];

    u_out[idx] = c + alpha * ((l + r) + (d + up) - 4.0f * c);
}