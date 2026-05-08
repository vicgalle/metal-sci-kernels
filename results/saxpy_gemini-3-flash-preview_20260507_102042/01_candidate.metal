#include <metal_stdlib>
using namespace metal;

/**
 * SAXPY: y[i] = a * x[i] + y[i]
 * Optimized for Apple Silicon by maximizing occupancy and using fused multiply-add.
 */
kernel void saxpy(
    device const float *x [[buffer(0)]],
    device float       *y [[buffer(1)]],
    constant float     &a [[buffer(2)]],
    constant uint      &N [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    // Standard boundary check. The host pads the grid, so this is necessary.
    if (i < N) {
        // Use the explicit fma intrinsic for y = a * x + y.
        // This ensures the compiler uses the most efficient hardware instruction
        // and helps maximize throughput on the unified memory architecture.
        y[i] = fma(x[i], a, y[i]);
    }
}