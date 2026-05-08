#include <metal_stdlib>
using namespace metal;

kernel void saxpy(device const float * restrict x [[buffer(0)]],
                  device float * restrict y [[buffer(1)]],
                  constant float &a [[buffer(2)]],
                  constant uint &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) {
    // Standard boundary check to handle dispatches padded to threadgroup size.
    if (i >= N) return;

    // Load constant 'a' once into a register.
    const float alpha = a;

    // Load x[i] and y[i] into registers to maximize Memory-Level Parallelism.
    // The restrict qualifier on the pointers tells the compiler that x and y
    // are independent, allowing it to issue these loads simultaneously.
    const float xi = x[i];
    const float yi = y[i];

    // Compute y = a * x + y using the fused multiply-add instruction.
    // This provides high throughput and precision in a single cycle.
    y[i] = fma(alpha, xi, yi);
}