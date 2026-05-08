#include <metal_stdlib>
using namespace metal;

kernel void saxpy(device const float *x [[buffer(0)]],
                  device float       *y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]])
{
    if (i >= N) return;
    // fma(a, x[i], y[i]) compiles to a single fused multiply-add instruction
    // on Apple Silicon, avoiding a separate load-mul-add sequence and
    // reducing register pressure vs. the naive a*x[i]+y[i] form.
    y[i] = fma(a, x[i], y[i]);
}