To improve performance beyond the incumbent, we revert the manual threadgroup memory tiling from the previous attempt—which incurred heavy barrier and address-math overhead—and trust Apple Silicon's massive L1/L2 caches to handle data reuse, since they already easily achieved ~222 GB/s on the naive version. 

Instead of memory, the bottleneck in the naive version is the integer ALU pipeline: it performs 5 separate 2D-to-1D multiplications (`j * NX + i`) per thread. We optimize this by computing a single base index (`idx`), and using simple addition/subtraction to fetch the top, bottom, left, and right neighbors. Pre-loading the center value also deduplicates memory access between the boundary copy and the interior calculation.

```metal
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
    uint nx = NX;
    uint ny = NY;

    if (i >= nx || j >= ny) return;

    // Single flat index simplifies address arithmetic for the whole stencil
    uint idx = j * nx + i;
    
    // Preload center value; handles both the boundary copy and the interior center
    float c = u_in[idx];

    // Dirichlet boundaries
    if (i == 0 || j == 0 || i == nx - 1 || j == ny - 1) {
        u_out[idx] = c;
        return;
    }

    // Relative offsets replace multiple integer multiplications per thread
    float l = u_in[idx - 1];
    float r = u_in[idx + 1];
    float d = u_in[idx - nx];
    float u = u_in[idx + nx];

    u_out[idx] = c + alpha * (l + r + d + u - 4.0f * c);
}
```