#include <metal_stdlib>
using namespace metal;

// Define the tile dimensions for threadgroup memory.
// Each threadgroup will process a tile of TILE_DIM_X * TILE_DIM_Y cells.
// We'll use a 16x16 tile, which results in 256 threads per threadgroup.
#define THREADGROUP_TILE_DIM_X 16
#define THREADGROUP_TILE_DIM_Y 16

// For the 5-point stencil, a 1-cell halo is needed around the tile in threadgroup memory.
// So, the actual threadgroup memory size will be (TILE_DIM_X + 2) * (TILE_DIM_Y + 2).
#define THREADGROUP_MEM_X (THREADGROUP_TILE_DIM_X + 2)
#define THREADGROUP_MEM_Y (THREADGROUP_TILE_DIM_Y + 2)

// Using max_total_threads_per_threadgroup helps the compiler optimize for the target hardware.
// For 16x16 threadgroup, total threads are 256.
kernel void heat_step(device const float *u_in  [[buffer(0)]],
                      device       float *u_out [[buffer(1)]],
                      constant uint      &NX    [[buffer(2)]],
                      constant uint      &NY    [[buffer(3)]],
                      constant float     &alpha [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]],
                      uint2 lid [[thread_position_in_threadgroup]],
                      uint2 group_id [[threadgroup_position_in_grid]])
                      [[max_total_threads_per_threadgroup(THREADGROUP_TILE_DIM_X * THREADGROUP_TILE_DIM_Y)]]
{
    // The dispatch grid is (NX, NY), so gid is the logical output cell coordinate.
    // The problem implies a 1:1 mapping of threads to output cells, but
    // to use threadgroup memory efficiently, we need to map gid to a threadgroup structure.
    // We achieve this by re-interpreting the gid as local to a tile computed by a threadgroup.

    // Calculate the global indices for the current thread's output cell within its group's tile.
    uint i = group_id.x * THREADGROUP_TILE_DIM_X + lid.x; // column
    uint j = group_id.y * THREADGROUP_TILE_DIM_Y + lid.y; // row

    // Early exit for threads outside the logical grid (NX, NY)
    if (i >= NX || j >= NY) {
        return;
    }

    // Dirichlet boundary conditions: copy boundary value through unchanged.
    if (i == 0 || j == 0 || i == NX - 1 || j == NY - 1) {
        u_out[j * NX + i] = u_in[j * NX + i];
        return;
    }

    // Declare threadgroup memory for the tile plus a 1-cell halo
    threadgroup float shared_tile[THREADGROUP_MEM_Y][THREADGROUP_MEM_X];

    // Calculate the global coordinates of the top-left corner of the
    // data block (including halo) that this threadgroup needs to load.
    // This is 1 cell to the top-left of the first cell this threadgroup computes.
    int global_src_x_start = (int)(group_id.x * THREADGROUP_TILE_DIM_X) - 1;
    int global_src_y_start = (int)(group_id.y * THREADGROUP_TILE_DIM_Y) - 1;

    // Load data from global memory into threadgroup memory.
    // Each thread cooperatively loads a part of the tile.
    // The loops ensure all (THREADGROUP_MEM_Y x THREADGROUP_MEM_X) cells are loaded.
    for (uint y_offset = lid.y; y_offset < THREADGROUP_MEM_Y; y_offset += THREADGROUP_TILE_DIM_Y) {
        for (uint x_offset = lid.x; x_offset < THREADGROUP_MEM_X; x_offset += THREADGROUP_TILE_DIM_X) {
            // Calculate global coordinates for the element being loaded
            int current_global_x = global_src_x_start + x_offset;
            int current_global_y = global_src_y_start + y_offset;

            // Clamp coordinates to stay within the original grid bounds for loading `u_in`.
            // This handles loading from regions near the global grid boundaries.
            // For Dirichlet conditions, values outside the grid effectively don't participate
            // in stencil computations; any value would be okay if it's never used.
            // But for cells *on* the actual boundary (e.g., global_y=0), we must load their value.
            uint clamped_global_x = metal::clamp((uint)current_global_x, 0u, NX - 1);
            uint clamped_global_y = metal::clamp((uint)current_global_y, 0u, NY - 1);

            // Load the value into threadgroup memory
            shared_tile[y_offset][x_offset] = u_in[clamped_global_y * NX + clamped_global_x];
        }
    }

    // Synchronize threads in the threadgroup to ensure all data is loaded before computation.
    threadgroup_barrier(mem_flags::threadgroup_mem_fence);

    // Compute the stencil using data from threadgroup memory.
    // The current thread (lid.x, lid.y) corresponds to index (lid.x + 1, lid.y + 1) in shared_tile
    // because the shared_tile has a 1-cell halo.
    uint sx = lid.x + 1; // Index into shared_tile for the current column
    uint sy = lid.y + 1; // Index into shared_tile for the current row

    float c = shared_tile[sy][sx];        // Center
    float l = shared_tile[sy][sx - 1];    // Left
    float r = shared_tile[sy][sx + 1];    // Right
    float d = shared_tile[sy - 1][sx];    // Down (j-1)
    float u = shared_tile[sy + 1][sx];    // Up (j+1)

    float result = c + alpha * (l + r + d + u - 4.0f * c);

    // Write the result back to global memory
    u_out[j * NX + i] = result;
}