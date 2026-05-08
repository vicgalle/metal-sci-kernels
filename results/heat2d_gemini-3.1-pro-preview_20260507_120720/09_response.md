To push performance past the memory-bound limit of the naïve kernel, we can use **SIMDgroup cooperation** to share memory loads across threads within the same warp via registers, significantly reducing L1 cache requests.

### Optimization Description
1. **SIMD-level Tiling:** We use `simd_shuffle_up` and `simd_shuffle_down` to exchange the center cell (`c`) with horizontal neighbors. For an internal cell, `l` and `r` are fetched directly from adjacent warp lanes almost for free, dropping redundant L1 load requests by ~40% and freeing up load/store units.
2. **Safe Full-Warp Shuffles:** To guarantee valid shuffled values, the shuffles are executed *before* any threads early-return. Invalid/out-of-bound threads are mapped to `idx = 0` so they still participate safely without reading unmapped memory. 
3. **Instruction-Level Parallelism:** The mathematical reduction is re-associated to `(l + r) + (d + u)`, which decreases the dependency chain depth and allows the GPU ALUs to compute the sums in parallel before feeding into a fused multiply-add (FMA).

```metal
#include <metal_stdlib>
using namespace metal;

kernel void heat_step(device const float *u_in  [[buffer(0)]],
                      device       float *u_out [[buffer(1)]],
                      constant uint      &NX    [[buffer(2)]],
                      constant uint      &NY    [[buffer(3)]],
                      constant float     &alpha [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]]) 
{
    uint nx = NX;
    uint ny = NY;
    
    uint i = gid.x;
    uint j = gid.y;
    
    // Evaluate validity but do NOT return early yet so all lanes can participate in the shuffles.
    // Invalid threads are safely pointed to idx 0 to prevent out-of-bounds memory accesses.
    bool valid = (i < nx && j < ny);
    uint idx = valid ? (j * nx + i) : 0;
    
    // Every thread loads its center value (invalid threads load u_in[0])
    float c = u_in[idx];
    
    // --- SIMDgroup Cooperation Phase ---
    uint lane = simd_lane_id;
    uint width = simd_execution_width;
    
    // Shuffle the linear index to robustly identify if adjacent lanes hold the true spatial neighbor
    uint l_idx = simd_shuffle_up(idx, 1);
    uint r_idx = simd_shuffle_down(idx, 1);
    
    // Shuffle the data values 
    float l_shfl = simd_shuffle_up(c, 1);
    float r_shfl = simd_shuffle_down(c, 1);
    
    // Now that shuffles are complete, invalid threads safely exit
    if (!valid) return;
    
    // Dirichlet boundary conditions: edge cells stay at their initial value.
    if (i == 0 || j == 0 || i == nx - 1 || j == ny - 1) {
        u_out[idx] = c;
        return;
    }
    
    // --- Internal Cells ONLY ---
    // Fetch horizontal neighbors from registers if available, else fallback to L1 cache.
    // Using explicit if/else ensures that the fallback load is strictly predicated.
    float l;
    if (lane > 0 && l_idx == idx - 1) {
        l = l_shfl;
    } else {
        l = u_in[idx - 1];
    }
    
    float r;
    if (lane < width - 1 && r_idx == idx + 1) {
        r = r_shfl;
    } else {
        r = u_in[idx + 1];
    }
    
    // Vertical neighbors must still be loaded from L1 cache (different rows).
    float d = u_in[idx - nx];
    float u = u_in[idx + nx];
    
    // FMA optimization with grouped additions to maximize instruction-level parallelism
    u_out[idx] = fma(alpha, (l + r) + (d + u) - 4.0f * c, c);
}
```