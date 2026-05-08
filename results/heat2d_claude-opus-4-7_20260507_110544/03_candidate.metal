#include <metal_stdlib>
using namespace metal;

kernel void heat_step(device const float *u_in  [[buffer(0)]],
                      device       float *u_out [[buffer(1)]],
                      constant uint      &NX    [[buffer(2)]],
                      constant uint      &NY    [[buffer(3)]],
                      constant float     &alpha [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]])
[[max_total_threads_per_threadgroup(256)]]
{
    uint i = gid.x;
    uint j = gid.y;
    if (i >= NX || j >= NY) return;

    uint idx = j * NX + i;
    float c = u_in[idx];

    bool is_boundary = (i == 0) | (j == 0) | (i == NX - 1) | (j == NY - 1);

    // Clamp neighbor indices so we can always load safely; the result is
    // discarded for boundary cells via the select() below.
    uint im1 = select(i - 1, i, i == 0);
    uint ip1 = select(i + 1, i, i == NX - 1);
    uint jm1 = select(j - 1, j, j == 0);
    uint jp1 = select(j + 1, j, j == NY - 1);

    float l = u_in[j * NX + im1];
    float r = u_in[j * NX + ip1];
    float d = u_in[jm1 * NX + i];
    float u = u_in[jp1 * NX + i];

    float stencil = c + alpha * ((l + r) + (d + u) - 4.0f * c);
    u_out[idx] = is_boundary ? c : stencil;
}