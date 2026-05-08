#include <metal_stdlib>
using namespace metal;

kernel void heat_step(device const float *u_in  [[buffer(0)]],
                      device       float *u_out [[buffer(1)]],
                      constant uint      &NX    [[buffer(2)]],
                      constant uint      &NY    [[buffer(3)]],
                      constant float     &alpha [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint j = gid.y;
    if (i >= NX || j >= NY) return;

    uint row    = j * NX;
    uint idx    = row + i;

    float c = u_in[idx];

    bool boundary = (i == 0) | (j == 0) | (i == NX - 1) | (j == NY - 1);
    if (boundary) {
        u_out[idx] = c;
        return;
    }

    float l = u_in[idx - 1];
    float r = u_in[idx + 1];
    float d = u_in[idx - NX];
    float uu = u_in[idx + NX];

    u_out[idx] = c + alpha * ((l + r) + (d + uu) - 4.0f * c);
}