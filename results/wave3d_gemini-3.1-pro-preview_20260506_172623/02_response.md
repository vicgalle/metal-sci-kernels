To optimize this memory-bound 3D stencil, we must maximize cache hit rates and hide memory latency by freeing up the ALU and reducing instruction dependency chains. The previous attempt tried using `simd_shuffle` to share X-axis neighbors, but on Apple Silicon's architecture, L1 coalescing already handles contiguous X loads perfectly. Adding SIMD shuffles introduced divergence and overhead, drastically reducing throughput. 

Instead, this version returns to straightforward, compiler-friendly memory loads but aggressively optimizes the math and control flow:
1. **ALU Reduction & FMA Chaining**: We mathematically rearrange the integration step. By precomputing the center scalar `c0 = 2.0 - 6.0 * alpha`, we can compute the entire update using exactly two chained FMA instructions (`fma(alpha, sum, fma(c0, c, -u_prev[idx]))`), cutting ALU operations and dependency depth.
2. **Tree Reduction**: The 6-neighbor sum is grouped into pairs (`(xm + xp) + (ym + yp) + ...`) to expose instruction-level parallelism (ILP).
3. **Control Flow**: The boundary checking uses unified integer checks without needing to evaluate the array strides first, ensuring boundary threads exit instantly before issuing any global loads besides their own value.

```metal
#include <metal_stdlib>
using namespace metal;

kernel void wave_step(device const float *u_prev [[buffer(0)]],
                      device const float *u_curr [[buffer(1)]],
                      device       float *u_next [[buffer(2)]],
                      constant uint      &NX     [[buffer(3)]],
                      constant uint      &NY     [[buffer(4)]],
                      constant uint      &NZ     [[buffer(5)]],
                      constant float     &alpha  [[buffer(6)]],
                      uint3 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint j = gid.y;
    uint k = gid.z;
    
    // Read constants to registers
    uint nx = NX;
    uint ny = NY;
    uint nz = NZ;
    
    // Bounds check
    if (i >= nx || j >= ny || k >= nz) return;

    uint idx = (k * ny + j) * nx + i;
    
    // Fetch center value first
    float c = u_curr[idx];

    // Dirichlet boundary conditions: edge cells copy u_curr unchanged
    if (i == 0 || j == 0 || k == 0 ||
        i == nx - 1 || j == ny - 1 || k == nz - 1) {
        u_next[idx] = c;
        return;
    }

    uint stride_y = nx;
    uint stride_z = nx * ny;

    // Load spatial neighbors (L1 cache handles spatial locality naturally)
    float xm = u_curr[idx - 1];
    float xp = u_curr[idx + 1];
    float ym = u_curr[idx - stride_y];
    float yp = u_curr[idx + stride_y];
    float zm = u_curr[idx - stride_z];
    float zp = u_curr[idx + stride_z];

    // Tree-reduce the sums to expose ILP and minimize dependency chain length
    float sum_x = xm + xp;
    float sum_y = ym + yp;
    float sum_z = zm + zp;
    float sum = (sum_x + sum_y) + sum_z;

    // Precompute the scalar multiple for the center cell to minimize ALU ops.
    // Original math: lap = sum - 6*c; next = 2*c - prev + alpha * lap
    // Refactored:    next = (2 - 6*alpha)*c + alpha*sum - prev
    float a = alpha;
    float c0 = 2.0f - 6.0f * a;
    
    // Compile down to 2 parallelizable FMA instructions
    u_next[idx] = fma(a, sum, fma(c0, c, -u_prev[idx]));
}
```