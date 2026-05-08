The kernel is optimized by reducing redundant index calculations and minimizing memory access overhead. By pre-calculating the linear index (`idx = j * NX + i`) and using offsets for the 5-point stencil (e.g., `idx - 1`, `idx + 1`, `idx - NX`, `idx + NX`), we reduce the number of multiplications per thread from six to one. This minimizes instruction pressure and simplifies address generation for the GPU's memory units. We utilize the `fma` (fused multiply-add) intrinsic to ensure the stencil update is performed in a single, high-precision cycle. Furthermore, by avoiding explicit threadgroup sizing, we allow the Metal driver to select the optimal dispatch configuration for Apple Silicon's unified memory architecture, which maximizes cache hit rates and throughput across different grid sizes.

```metal
#include <metal_stdlib>
using namespace metal;

kernel void heat_step(device const float *u_in  [[buffer(0)]],
                      device       float *u_out [[buffer(1)]],
                      constant uint      &NX    [[buffer(2)]],
                      constant uint      &NY    [[buffer(3)]],
                      constant float     &alpha [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x;
    const uint j = gid.y;

    // Standard grid boundary check
    if (i >= NX || j >= NY) return;

    const uint idx = j * NX + i;

    // Dirichlet Boundary Condition: Copy edges directly from input to output.
    // Early return helps avoid branching in the heavy stencil math for interior cells.
    if (i == 0 || j == 0 || i == NX - 1 || j == NY - 1) {
        u_out[idx] = u_in[idx];
        return;
    }

    // Load the 5-point stencil values.
    // Using index offsets from the pre-calculated 'idx' minimizes redundant multiplications.
    const float c = u_in[idx];
    const float l = u_in[idx - 1];
    const float r = u_in[idx + 1];
    const float u = u_in[idx - NX];
    const float d = u_in[idx + NX];

    // Discrete Laplacian: (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4*u[i,j])
    const float laplacian = (l + r + u + d) - 4.0f * c;

    // Heat Equation: u_new = u_old + alpha * Laplacian
    // fma(a, b, c) calculates (a * b + c) and is highly optimized on Apple Silicon.
    u_out[idx] = fma(alpha, laplacian, c);
}
```