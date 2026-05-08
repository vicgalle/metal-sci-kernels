#include <metal_stdlib>
using namespace metal;

kernel void saxpy(device const float * __restrict x [[buffer(0)]],
                  device float       * __restrict y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]])
{
    const uint n = N;
    const float aa = a;

    // Fast path for SIMD-width blocks that are entirely in bounds.
    if ((i | 31u) < n) {
        const float xi = x[i];
        const float yi = y[i];
        y[i] = fma(aa, xi, yi);
        return;
    }

    // Tail / padded-grid path.
    if (i < n) {
        const float xi = x[i];
        const float yi = y[i];
        y[i] = fma(aa, xi, yi);
    }
}