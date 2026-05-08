#include <metal_stdlib>
using namespace metal;

kernel void saxpy(device const float *x [[buffer(0)]],
                  device float       *y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) {
    // Structural optimization: Broadcast the constant multiplier 'a' across the SIMD-group.
    // Given the grid is padded to the threadgroup width (a multiple of 32),
    // all lanes in the SIMD-group participate in this broadcast.
    const float alpha = simd_broadcast((float)a, (ushort)0);
    
    // Standard boundary check to handle dispatches padded to threadgroup size.
    if (i >= N) return;

    // Load both input values into registers immediately.
    // This provides the compiler with clear intent to issue concurrent DRAM requests,
    // maximizing memory-level parallelism (MLP) to hide memory latency.
    const float xi = x[i];
    const float yi = y[i];

    // Use a single fused multiply-add (FMA) for optimal cycle count and precision.
    // On Apple Silicon, this is typically a single-cycle operation for the execution unit.
    y[i] = fma(alpha, xi, yi);
}