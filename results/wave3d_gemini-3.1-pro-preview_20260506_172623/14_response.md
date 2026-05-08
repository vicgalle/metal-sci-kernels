To improve upon the naive baseline, we must address instruction overhead and pipeline stalls rather than algorithmic restructuring, because the Apple L1 cache already near-optimally handles the spatial memory accesses for this tightly-coupled stencil.

1. **Uniform Preload & Branch Reduction:** The load for the center cell `c = u_curr[idx]` is hoisted *above* the boundary check. This guarantees uniform memory issue across the SIMD group before any lanes diverge. The boundary check itself is flattened using bitwise ORs (`|`) instead of logical ORs (`||`), eliminating cascaded short-circuit branches.
2. **Instruction Level Parallelism (ILP):** We explicitly decouple the calculation into `base = fma(2.0, c, -p)` and `lap = ...`. This allows the ALU to crunch the temporal update concurrently while waiting for the deep spatial memory accesses (`zm`, `zp`) to return.
3. **Compiler Hints & Aliasing:** Explicitly decorating the input and output buffer pointers with `restrict` guarantees to the compiler that `u_next` stores will never alias with `u_prev` or `u_curr` reads, unlocking more aggressive load-hoisting and scheduling.

```metal
#include <metal_stdlib>
using namespace metal;

kernel void wave_step(device const float * restrict u_prev [[buffer(0)]],
                      device const float * restrict u_curr [[buffer(1)]],
                      device       float * restrict u_next [[buffer(2)]],
                      constant uint      &NX     [[buffer(3)]],
                      constant uint      &NY     [[buffer(4)]],
                      constant uint      &NZ     [[buffer(5)]],
                      constant float     &alpha  [[buffer(6)]],
                      uint3 gid [[thread_position_in_grid]]) {
    uint nx = NX;
    uint ny = NY;
    uint nz = NZ;
    
    uint i = gid.x;
    uint j = gid.y;
    uint k = gid.z;

    if (i >= nx || j >= ny || k >= nz) return;

    uint stride_y = nx;
    uint stride_z = nx * ny;
    uint idx = k * stride_z + j * stride_y + i;

    // Hoist center load: primes the L1 cache for the entire SIMD group uniformly 
    // and satisfies the boundary copy requirement seamlessly.
    float c = u_curr[idx];

    // Evaluate boundary condition using bitwise ORs to avoid multiple 
    // short-circuit branches and warp divergence.
    bool is_boundary = (i == 0u) | (j == 0u) | (k == 0u) | 
                       (i == nx - 1u) | (j == ny - 1u) | (k == nz - 1u);

    if (is_boundary) {
        u_next[idx] = c;
        return;
    }

    // Load neighbors (coalesced naturally via L1 cache)
    float xm = u_curr[idx - 1u];
    float xp = u_curr[idx + 1u];
    float ym = u_curr[idx - stride_y];
    float yp = u_curr[idx + stride_y];
    float zm = u_curr[idx - stride_z];
    float zp = u_curr[idx + stride_z];

    // Load previous time step
    float p = u_prev[idx];

    float a = alpha;

    // Compute independent parts concurrently to maximize Instruction Level Parallelism
    float base = fma(2.0f, c, -p);
    float lap = xm + xp + ym + yp + zm + zp - 6.0f * c;

    // Final time integration
    u_next[idx] = fma(a, lap, base);
}
```