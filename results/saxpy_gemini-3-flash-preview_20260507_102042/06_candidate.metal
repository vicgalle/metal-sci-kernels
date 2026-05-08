#include <metal_stdlib>
using namespace metal;

kernel void saxpy(device const float *x [[buffer(0)]],
                  device float       *y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) [[max_total_threads_per_threadgroup(512)]] {
    // Standard boundary check to handle dispatches padded to threadgroup size.
    if (i >= N) return;

    // Capture the constant scalar and input values into registers.
    // Explicitly loading x[i] and y[i] into registers helps the compiler 
    // issue concurrent memory reads, improving Memory-Level Parallelism.
    const float alpha = a;
    const float xi = x[i];
    const float yi = y[i];

    // Compute y = a * x + y using the fused multiply-add (FMA) instruction.
    // On Apple Silicon, this is typically a high-throughput, single-cycle operation.
    y[i] = fma(alpha, xi, yi);
}