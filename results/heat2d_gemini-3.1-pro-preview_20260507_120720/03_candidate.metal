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
    
    // Bounds check for the grid dispatch padding
    if (i >= nx || j >= ny) return;

    // Single flat index for minimal address arithmetic
    uint idx = j * nx + i;

    // Read center element early; needed for both boundary copy and inner cell math
    float c = u_in[idx];

    // Dirichlet boundary condition via unsigned integer underflow.
    // Elegantly handles i == 0, i == nx - 1, j == 0, j == ny - 1 in fewer instructions.
    if ((i - 1) >= (nx - 2) || (j - 1) >= (ny - 2)) {
        u_out[idx] = c;
        return;
    }

    // Load neighbors in strictly monotonic memory order to maximize L1 prefetch efficiency
    float u = u_in[idx - nx];
    float l = u_in[idx - 1];
    float r = u_in[idx + 1];
    float d = u_in[idx + nx];
    
    float a = alpha;
    
    // Group neighbor additions into a depth-2 tree to maximize Instruction Level Parallelism
    float sum = (l + r) + (u + d);
    
    // Mathematically rewrite `c + alpha * (sum - 4*c)` to `c * (1 - 4*alpha) + alpha * sum`.
    // The uniform coefficient is evaluated once and hoisted by the compiler.
    float c_coeff = 1.0f - 4.0f * a;
    
    // Fused multiply-add directly on the critical path for optimal throughput
    u_out[idx] = fma(a, sum, c * c_coeff);
}