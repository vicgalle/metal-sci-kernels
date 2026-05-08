**Optimization:** The previous attempt failed because `[[max_total_threads_per_threadgroup(N)]]` must be placed before the function name (after the kernel return type), not after the parameter list. I'll fix that placement and keep the threadgroup tiling approach with a 16×16 tile + halo, which amortizes the 5 loads per cell down to ~1 load per cell from device memory. Cooperative halo loads use `clamp` to keep them safe even for OOB threads. This should significantly reduce DRAM traffic and improve bandwidth utilization, especially on the smaller sizes where the naive kernel is far below peak.

```metal
#include <metal_stdlib>
using namespace metal;

#define TS 16

kernel void heat_step(
    device const float *u_in  [[buffer(0)]],
    device       float *u_out [[buffer(1)]],
    constant uint      &NX    [[buffer(2)]],
    constant uint      &NY    [[buffer(3)]],
    constant float     &alpha [[buffer(4)]],
    uint2 gid  [[thread_position_in_grid]],
    uint2 lid  [[thread_position_in_threadgroup]])
{
    threadgroup float tile[TS + 2][TS + 2];

    int i = int(gid.x);
    int j = int(gid.y);
    int li = int(lid.x) + 1;
    int lj = int(lid.y) + 1;

    int NXi = int(NX);
    int NYi = int(NY);

    // Center load (clamped so halo writes are always safe).
    int ci = clamp(i, 0, NXi - 1);
    int cj = clamp(j, 0, NYi - 1);
    tile[lj][li] = u_in[cj * NXi + ci];

    // Halo: left/right columns
    if (lid.x == 0) {
        int hi = clamp(i - 1, 0, NXi - 1);
        tile[lj][0] = u_in[cj * NXi + hi];
    }
    if (lid.x == TS - 1) {
        int hi = clamp(i + 1, 0, NXi - 1);
        tile[lj][TS + 1] = u_in[cj * NXi + hi];
    }
    // Halo: top/bottom rows
    if (lid.y == 0) {
        int hj = clamp(j - 1, 0, NYi - 1);
        tile[0][li] = u_in[hj * NXi + ci];
    }
    if (lid.y == TS - 1) {
        int hj = clamp(j + 1, 0, NYi - 1);
        tile[TS + 1][li] = u_in[hj * NXi + ci];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (i >= NXi || j >= NYi) return;

    if (i == 0 || j == 0 || i == NXi - 1 || j == NYi - 1) {
        u_out[j * NXi + i] = tile[lj][li];
        return;
    }

    float c = tile[lj][li];
    float l = tile[lj][li - 1];
    float r = tile[lj][li + 1];
    float d = tile[lj - 1][li];
    float uu = tile[lj + 1][li];
    u_out[j * NXi + i] = c + alpha * (l + r + d + uu - 4.0f * c);
}
```