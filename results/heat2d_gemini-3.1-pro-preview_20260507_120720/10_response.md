I am applying register-level and instruction-level optimizations to the seed kernel to maximize throughput without introducing the overhead of threadgroup barriers or SIMD-shuffle branches. 

1. **Index Math Reduction:** Redundant 2D index calculations like `(j - 1) * NX + i` are replaced with direct 1D offsets (`idx - nx`), explicitly minimizing ALU instructions that the compiler's Common Subexpression Elimination might miss.
2. **Unified Center Load:** The center element `c = u_in[idx]` is hoisted above the boundary check. This ensures a unified, non-divergent memory fetch for the entire SIMD group before edge threads branch and return.
3. **ILP & FMA:** The neighbor additions are grouped into a balanced tree `(l + r) + (d + u)` to expose Instruction-Level Parallelism and reduce dependency latency, concluding with a fused multiply-add (`fma`).

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

    uint idx = j * nx + i;
    
    // Load center value immediately; this is used by both boundary and interior cells.
    // Hoisting this load guarantees uniform memory access across the SIMD group.
    float c = u_in[idx];

    // Dirichlet boundary conditions: edge cells stay at their initial value.
    if (i == 0 || j == 0 || i == nx - 1 || j == ny - 1) {
        u_out[idx] = c;
        return;
    }

    // Fetch neighbors using simple offset arithmetic rather than full 2D index recalculation.
    // This strictly reduces integer ALU operations.
    float l = u_in[idx - 1];
    float r = u_in[idx + 1];
    float d = u_in[idx - nx];
    float u = u_in[idx + nx];

    // Grouped additions `(l + r) + (d + u)` reduce latency via instruction-level parallelism.
    // The FMA intrinsic efficiently fuses the final scaling and addition.
    u_out[idx] = fma(alpha, (l + r) + (d + u) - 4.0f * c, c);
}
```