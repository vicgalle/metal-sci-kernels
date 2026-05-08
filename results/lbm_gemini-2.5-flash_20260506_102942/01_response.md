This optimization applies threadgroup memory tiling to reduce global memory bandwidth. The D2Q9 LBM pull-streaming step accesses 9 neighboring cells. By loading a block of input data plus a 1-cell halo into threadgroup memory, each required `f_in` value from global memory is read only once by the threadgroup, then reused by multiple threads from fast on-chip memory. This reduces redundant global memory reads for overlapping halo regions between cells. Additionally, the expensive modulo operations for periodic boundary conditions are replaced with faster conditional arithmetic for small offsets.

```metal
#include <metal_stdlib>
using namespace metal;

// Velocity directions:
//   0: ( 0, 0)   1: (+1, 0)   2: ( 0,+1)   3: (-1, 0)   4: ( 0,-1)
//   5: (+1,+1)   6: (-1,+1)   7: (-1,-1)   8: (+1,-1)
constant int CX[9] = {0,  1,  0, -1,  0,  1, -1, -1,  1};
constant int CY[9] = {0,  0,  1,  0, -1,  1,  1, -1, -1};
constant float W[9] = {
    4.0f / 9.0f,
    1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
    1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f
};

// Threadgroup block dimensions
// Must be power-of-2 for good SIMD group alignment.
// 16x16 = 256 threads per threadgroup.
#define BLOCK_X 16
#define BLOCK_Y 16
#define PADDED_BLOCK_X (BLOCK_X + 2)
#define PADDED_BLOCK_Y (BLOCK_Y + 2)

kernel void lbm_step(device const float *f_in   [[buffer(0)]],
                     device       float *f_out  [[buffer(1)]],
                     constant uint        &NX   [[buffer(2)]],
                     constant uint        &NY   [[buffer(3)]],
                     constant float       &tau  [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]],
                     uint2 local_id [[thread_position_in_threadgroup]],
                     uint2 group_id [[threadgroup_position_in_grid]]) {

    // Guard against out-of-bounds work-items (if grid is not multiple of BLOCK_X/Y)
    if (gid.x >= NX || gid.y >= NY) return;

    // Threadgroup memory for current block + 1-cell halo for each of 9 distributions
    threadgroup float tg_f[9][PADDED_BLOCK_Y][PADDED_BLOCK_X];

    // Global base coordinates for the current threadgroup's block
    int block_base_x = group_id.x * BLOCK_X;
    int block_base_y = group_id.y * BLOCK_Y;

    // Each thread loads values into the padded threadgroup memory.
    // We cover the (BLOCK_X+2) x (BLOCK_Y+2) region.
    // Each thread takes one point in the padded region and loads its 9 distributions.
    // Total padded elements: PADDED_BLOCK_X * PADDED_BLOCK_Y
    // Total threads: BLOCK_X * BLOCK_Y
    // Each thread will load N = ceil(PADDED_BLOCK_X * PADDED_BLOCK_Y / (BLOCK_X * BLOCK_Y)) elements
    // For 16x16 block: 18x18=324 elements to load, 256 threads. Each thread loads 2 elements.
    uint lidx_flat = local_id.y * BLOCK_X + local_id.x;
    for (uint tile_elem_idx = lidx_flat; tile_elem_idx < PADDED_BLOCK_X * PADDED_BLOCK_Y; tile_elem_idx += BLOCK_X * BLOCK_Y) {
        uint tx_padded = tile_elem_idx % PADDED_BLOCK_X;
        uint ty_padded = tile_elem_idx / PADDED_BLOCK_X;

        // Calculate global coordinates for the current element in padded tile
        int gx = block_base_x + int(tx_padded) - 1;
        int gy = block_base_y + int(ty_padded) - 1;

        // Apply periodic boundary conditions to get actual source coordinates
        // Using conditional arithmetic instead of modulo for small offsets
        if (gx < 0) gx += NX;
        else if (gx >= NX) gx -= NX;

        if (gy < 0) gy += NY;
        else if (gy >= NY) gy -= NY;

        // Load 9 distribution functions for this (gx, gy) cell into threadgroup memory
        uint N = NX * NY;
        uint src_idx = uint(gy) * NX + uint(gx);
        #pragma clang loop unroll(full)
        for (int k = 0; k < 9; ++k) {
            tg_f[k][ty_padded][tx_padded] = f_in[uint(k) * N + src_idx];
        }
    }

    // Synchronize to ensure all data is loaded into threadgroup memory
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Current thread's local (relative to block) and global coordinates
    int ltx = int(local_id.x);
    int lty = int(local_id.y);
    uint idx = gid.y * NX + gid.x;

    // 1. Pull streaming (from threadgroup memory)
    float f_streamed[9];
    #pragma clang loop unroll(full)
    for (int k = 0; k < 9; ++k) {
        // Source coordinates relative to current thread's position in padded tg_f
        // e.g., for CX[k]=1, need f from ltx+1 (local_id.x - 1)
        int src_ltx_padded = ltx - CX[k] + 1; // +1 for halo offset
        int src_lty_padded = lty - CY[k] + 1; // +1 for halo offset
        f_streamed[k] = tg_f[k][src_lty_padded][src_ltx_padded];
    }

    // 2. Moments.
    float rho = 0.0f;
    float ux  = 0.0f;
    float uy  = 0.0f;
    #pragma clang loop unroll(full)
    for (int k = 0; k < 9; ++k) {
        rho += f_streamed[k];
        ux  += float(CX[k]) * f_streamed[k];
        uy  += float(CY[k]) * f_streamed[k];
    }
    float inv_rho = 1.0f / rho;
    ux *= inv_rho;
    uy *= inv_rho;

    // 3. BGK collision.
    float usq = ux * ux + uy * uy;
    float inv_tau = 1.0f / tau;
    uint  N    = NX * NY;

    #pragma clang loop unroll(full)
    for (int k = 0; k < 9; ++k) {
        float cu  = float(CX[k]) * ux + float(CY[k]) * uy;
        float feq = W[k] * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * usq);
        f_out[uint(k) * N + idx] = f_streamed[k] - inv_tau * (f_streamed[k] - feq);
    }
}
```