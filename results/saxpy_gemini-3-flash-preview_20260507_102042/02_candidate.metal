#include <metal_stdlib>
using namespace metal;

kernel void saxpy(device const float *x [[buffer(0)]],
                  device float       *y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) {
    // Standard boundary check to handle padded grid dispatches.
    if (i >= N) return;

    // Loading values into local variables first to encourage the compiler 
    // to issue both loads in parallel, improving memory-level parallelism (MLP).
    float xi = x[i];
    float yi = y[i];

    // Standard multiply-add syntax. On Apple Silicon, this typically maps to a single
    // cycle MAD instruction. The scalar 'a' is uniform across the SIMD group,
    // allowing for efficient register use.
    y[i] = a * xi + yi;
}