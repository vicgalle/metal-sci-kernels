// Naive seed kernel for saxpy: y[i] = a * x[i] + y[i].
//
// Buffer layout (must be preserved by candidate kernels):
//   buffer 0: const float* x   (read-only, length N)
//   buffer 1: device float* y  (read-write, length N)
//   buffer 2: const float& a   (scalar)
//   buffer 3: const uint& N    (length)

#include <metal_stdlib>
using namespace metal;

kernel void saxpy(device const float *x [[buffer(0)]],
                  device float       *y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) {
    if (i >= N) return;
    y[i] = a * x[i] + y[i];
}
