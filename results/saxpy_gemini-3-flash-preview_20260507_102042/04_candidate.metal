#include <metal_stdlib>
using namespace metal;

// Hint the maximum threadgroup size to optimize occupancy and register pressure.
// 256 is a standard "sweet spot" for many Apple Silicon GPU architectures.
[[max_total_threads_per_threadgroup(256)]]
kernel void saxpy(device const float *x [[buffer(0)]],
                  device float       *y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) {
    // Standard boundary check to handle dispatches padded to threadgroup size.
    if (i < N) {
        // Loading into local variables encourages the compiler to issue concurrent 
        // memory loads (Memory-Level Parallelism), saturating the unified memory bus.
        const float xi = x[i];
        const float yi = y[i];

        // fma() ensures a single-cycle fused multiply-add instruction is used,
        // which is the most efficient way to compute a*x + y on Apple Silicon.
        y[i] = fma(xi, a, yi);
    }
}