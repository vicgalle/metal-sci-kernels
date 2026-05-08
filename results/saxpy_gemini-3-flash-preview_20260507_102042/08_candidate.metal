#include <metal_stdlib>
using namespace metal;

kernel void saxpy(
    device const float *x [[buffer(0)]],
    device float       *y [[buffer(1)]],
    constant float     &a [[buffer(2)]],
    constant uint      &N [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) [[max_total_threads_per_threadgroup(1024)]] {
    // Check bounds to handle cases where the grid is padded to the threadgroup size.
    if (i >= N) return;

    // Explicitly loading values into registers encourages the hardware to 
    // issue independent memory requests, maximizing bandwidth utilization.
    const float xi = x[i];
    const float yi = y[i];

    // The fused multiply-add (fma) instruction is highly optimized on Apple Silicon,
    // performing the operation y = a * x + y with a single instruction.
    // Using the 'a' reference directly in the fma call helps the compiler 
    // schedule the constant buffer load efficiently.
    y[i] = fma(a, xi, yi);
}