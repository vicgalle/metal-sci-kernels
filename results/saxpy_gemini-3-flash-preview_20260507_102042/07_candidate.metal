#include <metal_stdlib>
using namespace metal;

kernel void saxpy(device const float *x [[buffer(0)]],
                  device float       *y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) {
    // Early exit for threads dispatched beyond the array bounds due to padding.
    if (i >= N) return;

    // Cache the scalar constant and input values into registers.
    // This helps the compiler schedule independent loads simultaneously,
    // improving memory-level parallelism (MLP) on memory-bound workloads.
    const float alpha = a;
    const float xi = x[i];
    const float yi = y[i];

    // Compute y = a * x + y using the fused multiply-add intrinsic.
    // On Apple Silicon, this is a single-cycle high-throughput instruction.
    y[i] = fma(alpha, xi, yi);
}