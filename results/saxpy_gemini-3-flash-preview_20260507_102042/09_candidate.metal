#include <metal_stdlib>
using namespace metal;

kernel void saxpy(device const float *x [[buffer(0)]],
                  device float       *y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) {
    // Boundary check to ensure we don't access memory out of bounds
    // when the grid size is rounded up to the threadgroup size.
    if (i < N) {
        // Load the multiplier 'a' into a register once per thread.
        const float alpha = a;
        
        // Reading x[i] and y[i] into local registers allows the hardware
        // to issue these memory requests in parallel.
        const float xi = x[i];
        const float yi = y[i];
        
        // Perform the calculation y = a * x + y using the fused multiply-add 
        // instruction, which is typically a single-cycle operation on 
        // Apple Silicon GPUs and minimizes precision loss.
        y[i] = fma(alpha, xi, yi);
    }
}