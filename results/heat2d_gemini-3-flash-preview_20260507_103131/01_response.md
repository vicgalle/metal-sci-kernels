The optimization uses **threadgroup memory (tiling)** to reduce global memory bandwidth. By loading a block of the grid into shared memory, we reduce the number of redundant loads for the 5-point stencil (from 5 per thread to approximately 1.2–1.5 per thread). We use **cooperative halo loading** where only the threads at the edges of a threadgroup fetch neighbor values from adjacent blocks. This approach effectively uses the GPU's low-latency shared memory and reduces the pressure on the L2 cache and VRAM. The implementation uses a 1D tile array to handle arbitrary threadgroup sizes (up to 1024 threads) robustly. The Dirichlet boundary condition is handled by checking grid edges before stencil computation, ensuring that only internal cells are updated.

```metal
#include <metal_stdlib>
using namespace metal;

kernel void heat_step(device const float *u_in  [[buffer(0)]],
                      device       float *u_out [[buffer(1)]],
                      constant uint      &NX    [[buffer(2)]],
                      constant uint      &NY    [[buffer(3)]],
                      constant float     &alpha [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]],
                      uint2 lid [[thread_position_in_threadgroup]],
                      uint2 lsize [[threads_per_threadgroup]]) {
    const int i = (int)gid.x;
    const int j = (int)gid.y;
    const int nx = (int)NX;
    const int ny = (int)NY;
    const int lx = (int)lid.x;
    const int ly = (int)lid.y;
    const int sw = (int)lsize.x;
    const int sh = (int)lsize.y;
    const int stride = sw + 2;

    // Threadgroup memory to store the tile plus a 1-pixel halo.
    // 4096 floats = 16KB, which is well within the 32KB limit on Apple Silicon.
    // This size supports any configuration where (sw+2)*(sh+2) <= 4096.
    threadgroup float tile[4096];

    const bool in_bounds = (i < nx && j < ny);
    const int idx = j * nx + i;
    const int tile_idx = (ly + 1) * stride + (lx + 1);

    if (in_bounds) {
        // Every thread loads its central value into the tile.
        const float center_val = u_in[idx];
        tile[tile_idx] = center_val;

        // Cooperative halo loading: only threads on the edge of the threadgroup 
        // load neighbor values that are outside the current block.
        if (lx == 0 && i > 0) 
            tile[tile_idx - 1] = u_in[idx - 1];
        if (lx == sw - 1 && i < nx - 1) 
            tile[tile_idx + 1] = u_in[idx + 1];
        if (ly == 0 && j > 0) 
            tile[tile_idx - stride] = u_in[idx - nx];
        if (ly == sh - 1 && j < ny - 1) 
            tile[tile_idx + stride] = u_in[idx + nx];
    }

    // Synchronize to ensure all threads in the group have finished loading.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!in_bounds) return;

    // Dirichlet Boundary Conditions: Edge cells copy initial value and return.
    if (i == 0 || j == 0 || i == nx - 1 || j == ny - 1) {
        u_out[idx] = tile[tile_idx];
    } else {
        // Access 5-point stencil values from the high-speed shared tile.
        const float c = tile[tile_idx];
        const float l = tile[tile_idx - 1];
        const float r = tile[tile_idx + 1];
        const float u = tile[tile_idx - stride];
        const float d = tile[tile_idx + stride];

        // u_new = u + alpha * Laplacian
        u_out[idx] = c + alpha * (l + r + u + d - 4.0f * c);
    }
}
```