This optimization uses **threadgroup-memory tiling** to reduce global memory bandwidth. By loading a block of the input grid (including a 1-pixel "apron") into high-speed `threadgroup` memory, we can reuse values for the 5-point stencil across all threads in the group. For a standard 32x8 threadgroup, this reduces the number of global memory reads per cell from 5 to approximately 1.3. On Apple Silicon's unified memory architecture, reducing memory pressure is critical for stencil operations, which are traditionally memory-bound. We also use the `fma` (fused multiply-add) intrinsic to further optimize the discrete Laplacian calculation.

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
                      uint2 tsize [[threads_per_threadgroup]]) {
    const uint i = gid.x;
    const uint j = gid.y;

    // Tile storage for threadgroup-level data reuse.
    // Supports threadgroups up to 32x32 (though 256 total threads is recommended).
    threadgroup float tile[34][34];

    const uint ti = lid.x;
    const uint tj = lid.y;

    if (i < NX && j < NY) {
        const uint idx = j * NX + i;

        // Cooperative load: each thread fetches its central value.
        tile[tj + 1][ti + 1] = u_in[idx];

        // Load the 1-pixel apron (Left, Right, Up, Down). 
        // Corners are not required for a 5-point stencil.
        if (ti == 0 && i > 0) {
            tile[tj + 1][0] = u_in[idx - 1];
        }
        if (ti == tsize.x - 1 && i < NX - 1) {
            tile[tj + 1][ti + 2] = u_in[idx + 1];
        }
        if (tj == 0 && j > 0) {
            tile[0][ti + 1] = u_in[idx - NX];
        }
        if (tj == tsize.y - 1 && j < NY - 1) {
            tile[tj + 2][ti + 1] = u_in[idx + NX];
        }

        // Synchronize threads to ensure the tile is fully loaded.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Dirichlet Boundary Condition: check if the cell is on the grid edge.
        if (i > 0 && j > 0 && i < NX - 1 && j < NY - 1) {
            const float c = tile[tj + 1][ti + 1];
            const float l = tile[tj + 1][ti];
            const float r = tile[tj + 1][ti + 2];
            const float u = tile[tj][ti + 1];
            const float d = tile[tj + 2][ti + 1];

            // Discrete Laplacian and update using optimized math.
            // u_new = u + alpha * (L + R + U + D - 4*u)
            const float laplacian = (l + r + u + d) - 4.0f * c;
            u_out[idx] = fma(alpha, laplacian, c);
        } else {
            // Edges remain unchanged (Dirichlet BC).
            u_out[idx] = tile[tj + 1][ti + 1];
        }
    }
}
```