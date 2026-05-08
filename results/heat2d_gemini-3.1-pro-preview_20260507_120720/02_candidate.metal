#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(1024)]]
kernel void heat_step(device const float *u_in  [[buffer(0)]],
                      device       float *u_out [[buffer(1)]],
                      constant uint      &NX_ref    [[buffer(2)]],
                      constant uint      &NY_ref    [[buffer(3)]],
                      constant float     &alpha_ref [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint j = gid.y;
    uint NX = NX_ref;
    uint NY = NY_ref;
    
    // Bounds check for the grid dispatch padding
    if (i >= NX || j >= NY) return;

    // Single flat index for minimal address arithmetic
    uint idx = j * NX + i;

    // Read center element unconditionally to hide latency
    float c = u_in[idx];

    // Dirichlet boundary condition: boundary cells stay at their initial value.
    // Uses integer underflow to combine (i == 0 || i == NX - 1) into a single check.
    if ((i - 1) >= (NX - 2) || (j - 1) >= (NY - 2)) {
        u_out[idx] = c;
        return;
    }

    // Load neighbors (perfectly cached by L1)
    float l = u_in[idx - 1];
    float r = u_in[idx + 1];
    float u = u_in[idx - NX];
    float d = u_in[idx + NX];
    
    float alpha = alpha_ref;
    
    // Group neighbor additions into a depth-2 tree for ILP
    float sum = (l + r) + (u + d);
    
    // Mathematically rewrite `c + alpha * (sum - 4*c)` to remove one FP operation
    // from the critical path. The uniform value `1 - 4*alpha` is hoisted by the compiler.
    float c_coeff = 1.0f - 4.0f * alpha;
    
    // Fused multiply-add for optimal throughput and accuracy
    u_out[idx] = fma(alpha, sum, c * c_coeff);
}