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

kernel void saxpy(device const float *x [[buffer(0)]],
                  device float       *y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) {
    // Standard boundary check to handle dispatches padded to threadgroup size.
    if (i >= N) return;

    // Loading values into registers first encourages the compiler to issue 
    // concurrent loads for x[i] and y[i], improving Memory-Level Parallelism.
    const float xi = x[i];
    const float yi = y[i];

    // Use fused multiply-add to compute y = a * x + y in a single operation.
    // This is typically a single-cycle instruction on Apple Silicon GPUs.
    y[i] = fma(a, xi, yi);
}
```

Result of previous attempt:
                1M: correct, 0.09 ms, 144.1 GB/s (72.1% of 200 GB/s)
               16M: correct, 1.19 ms, 169.7 GB/s (84.9% of 200 GB/s)
               64M: correct, 4.31 ms, 187.0 GB/s (93.5% of 200 GB/s)
  score (gmean of fraction): 0.8301

## History

- iter  0: compile=OK | correct=True | score=0.8135375071068955
- iter  1: compile=OK | correct=True | score=0.8051676453474895
- iter  2: compile=OK | correct=True | score=0.8184898108134617
- iter  3: compile=OK | correct=True | score=0.8301040861563588

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
