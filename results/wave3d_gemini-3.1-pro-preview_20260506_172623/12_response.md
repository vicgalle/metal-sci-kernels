To optimize this memory-bound 3D wave equation kernel, we remove the `simd_shuffle` overhead from the previous attempt. Since threads in a warp access contiguous `X` elements, X-neighbor loads are already perfectly handled by the hardware L1 cache; manually shuffling them only adds ALU and synchronization latency without saving main memory bandwidth. 

Instead, this version focuses on maximizing memory-level parallelism (MLP) and ALU efficiency:
1. **`restrict` Pointers:** Added to guarantee no aliasing between the input and output buffers, allowing aggressive compiler reordering.
2. **Load Scheduling:** Center cell is loaded early to resolve the boundary condition. For internal cells, far-away Z and Y neighbors are fetched first to initiate long-latency cache misses as early as possible.
3. **ALU Optimization:** The 7-point stencil is computed using an independent tree-reduction for additions. The time-integration formula is algebraically simplified to use Fused-Multiply-Add (FMA) instructions, saving several math operations per cell and minimizing data-dependency chains.

```metal
#include <metal_stdlib>
using namespace metal;

kernel void wave_step(device const float* restrict u_prev [[buffer(0)]],
                      device const float* restrict u_curr [[buffer(1)]],
                      device       float* restrict u_next [[buffer(2)]],
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

    if (i >= nx || j >= ny || k >= nz) {
        return;
    }

    uint stride_y = nx;
    uint stride_z = nx * ny;
    uint idx = k * stride_z + j * stride_y + i;

    // Load center early. This is a highly local memory access and immediately
    // resolves the value needed for boundary cells.
    float c = u_curr[idx];

    if (i == 0 || j == 0 || k == 0 || i == nx - 1 || j == ny - 1 || k == nz - 1) {
        // Dirichlet: boundary cells stay put (u_next == u_curr).
        u_next[idx] = c;
        return;
    }

    // Issue far-away memory loads first (Z and Y neighbors) to maximize MLP.
    // These cross large memory strides and are most likely to incur cache misses.
    float zm = u_curr[idx - stride_z];
    float zp = u_curr[idx + stride_z];
    
    float ym = u_curr[idx - stride_y];
    float yp = u_curr[idx + stride_y];
    
    // Issue near loads (X neighbors and previous timestep)
    float xm = u_curr[idx - 1];
    float xp = u_curr[idx + 1];
    
    float p  = u_prev[idx];

    // Tree reduction for spatial additions to minimize dependency chain latency
    float z_sum = zm + zp;
    float y_sum = ym + yp;
    float x_sum = xm + xp;
    float lap_sum = z_sum + y_sum + x_sum;

    // Algebraic simplification for the time integration:
    // u_next = 2*c - p + alpha * (lap_sum - 6*c)
    //        = alpha * lap_sum + (2 - 6*alpha)*c - p
    float a = alpha;
    float c_coeff = 2.0f - 6.0f * a;

    u_next[idx] = fma(a, lap_sum, fma(c_coeff, c, -p));
}
```