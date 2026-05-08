(1) Optimization: Use `simd_broadcast` to share the scalar multiplier `a` across the SIMD-group (32 threads). This reduces redundant constant-cache access by ensuring the value is fetched once per group and then distributed via fast register shuffles. (2) Maximize Memory-Level Parallelism (MLP): Capture both `x[i]` and `y[i]` into local registers before the `fma` instruction to encourage the hardware's load-store units to issue both memory requests in parallel.

(2) Why: For memory-bound kernels like SAXPY on Apple Silicon, performance is limited by the number of outstanding memory requests and the efficiency of constant-buffer access. Broadcasting constants reduces the pressure on the scalar units, and register-staging the inputs ensures that the unified memory controller can saturate the bus by overlapping the read cycles of both buffers.

```metal
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
```