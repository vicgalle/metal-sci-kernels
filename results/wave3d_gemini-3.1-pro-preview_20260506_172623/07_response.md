(1) Optimization: Vectorized loads for contiguous X elements (`packed_float3`), a balanced addition tree to improve Instruction-Level Parallelism (ILP), and reordered memory operations to dispatch `u_prev` earlier. Boundary checks are consolidated using `min()` to reduce branching.
(2) Why it improves: Fetching the X-axis neighborhood via a single `packed_float3` load reduces load instructions and memory port pressure. The balanced addition tree cuts the sequential dependency depth of the Laplacian sum from 6 cycles down to 3, helping hide memory latency. Early load dispatch allows the GPU to overlap `u_prev` and `u_curr` requests more effectively. Consolidating the boundary condition avoids multiple conditional branches.

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

    if (i >= NX || j >= NY || k >= NZ) return;

    uint idx = (k * NY + j) * NX + i;

    // Dirichlet boundary condition: consolidate checks to reduce branching
    if (min(min(i, j), k) == 0 || i == NX - 1 || j == NY - 1 || k == NZ - 1) {
        u_next[idx] = u_curr[idx];
        return;
    }

    uint stride_y = NX;
    uint stride_z = NX * NY;

    // Issue u_prev load early to hide latency
    float p = u_prev[idx];

    // Issue u_curr Z and Y loads
    float zm = u_curr[idx - stride_z];
    float ym = u_curr[idx - stride_y];
    
    // Vector load for contiguous X elements (xm, c, xp) to save load instructions
    float3 x_vec = float3(*(device const packed_float3*)(&u_curr[idx - 1u]));
    float xm = x_vec.x;
    float c  = x_vec.y;
    float xp = x_vec.z;

    float yp = u_curr[idx + stride_y];
    float zp = u_curr[idx + stride_z];

    // Compute Laplacian with a balanced addition tree for better ILP
    float lap = ((xm + xp) + (ym + yp)) + (zm + zp) - 6.0f * c;
    
    // Final leapfrog update
    u_next[idx] = (2.0f * c - p) + alpha * lap;
}
```