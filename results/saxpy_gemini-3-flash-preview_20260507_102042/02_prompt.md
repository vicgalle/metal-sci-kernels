## Task: saxpy

SAXPY: out-of-place y = a*x + y. Memory-bound; expected to be BW-bound on Apple Silicon. Bytes moved per element = 12 (load x, load y, store y).

## Required kernel signature(s)

```
kernel void saxpy(device const float *x [[buffer(0)]],
                  device float       *y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]]);

Update y[i] = a * x[i] + y[i] for i in [0, N). Threads are dispatched 1-D, one per element (grid is padded up to a multiple of the threadgroup width, so guard against i >= N). Each thread MUST handle exactly one i; the host will not shrink the dispatch if you process multiple elements per thread — extra threads just idle.
```

## Your previous attempt

```metal
#include <metal_stdlib>
using namespace metal;

/**
 * SAXPY: y[i] = a * x[i] + y[i]
 * Optimized for Apple Silicon by maximizing occupancy and using fused multiply-add.
 */
kernel void saxpy(
    device const float *x [[buffer(0)]],
    device float       *y [[buffer(1)]],
    constant float     &a [[buffer(2)]],
    constant uint      &N [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    // Standard boundary check. The host pads the grid, so this is necessary.
    if (i < N) {
        // Use the explicit fma intrinsic for y = a * x + y.
        // This ensures the compiler uses the most efficient hardware instruction
        // and helps maximize throughput on the unified memory architecture.
        y[i] = fma(x[i], a, y[i]);
    }
}
```

Result of previous attempt:
                1M: correct, 0.09 ms, 144.5 GB/s (72.3% of 200 GB/s)
               16M: correct, 1.26 ms, 159.8 GB/s (79.9% of 200 GB/s)
               64M: correct, 4.45 ms, 180.8 GB/s (90.4% of 200 GB/s)
  score (gmean of fraction): 0.8052

## Current best (incumbent)

```metal
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
```

Incumbent result:
                1M: correct, 0.09 ms, 144.9 GB/s (72.5% of 200 GB/s)
               16M: correct, 1.26 ms, 160.3 GB/s (80.2% of 200 GB/s)
               64M: correct, 4.34 ms, 185.4 GB/s (92.7% of 200 GB/s)
  score (gmean of fraction): 0.8135

## History

- iter  0: compile=OK | correct=True | score=0.8135375071068955
- iter  1: compile=OK | correct=True | score=0.8051676453474895

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
