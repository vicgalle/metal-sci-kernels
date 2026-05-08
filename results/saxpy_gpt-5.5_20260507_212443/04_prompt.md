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

kernel void saxpy(device const float * __restrict x [[buffer(0)]],
                  device float       * __restrict y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]])
{
    const uint n = N;
    if (i >= n) {
        return;
    }

    const float aa = a;
    device float * __restrict yp = y + i;

    const float xi = x[i];
    const float yi = *yp;
    *yp = fma(aa, xi, yi);
}
```

Result of previous attempt:
                1M: correct, 0.09 ms, 144.0 GB/s (72.0% of 200 GB/s)
               16M: correct, 1.26 ms, 159.7 GB/s (79.8% of 200 GB/s)
               64M: correct, 4.51 ms, 178.6 GB/s (89.3% of 200 GB/s)
  score (gmean of fraction): 0.8006

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

kernel void saxpy(device const float * __restrict x [[buffer(0)]],
                  device float       * __restrict y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) {
    const uint n = N;
    const float aa = a;

    if (i < n) {
        const float xi = x[i];
        const float yi = y[i];
        y[i] = fma(aa, xi, yi);
    }
}
```

Incumbent result:
                1M: correct, 0.09 ms, 143.9 GB/s (72.0% of 200 GB/s)
               16M: correct, 1.25 ms, 160.5 GB/s (80.3% of 200 GB/s)
               64M: correct, 4.35 ms, 185.0 GB/s (92.5% of 200 GB/s)
  score (gmean of fraction): 0.8114

## History

- iter  0: compile=OK | correct=True | score=0.8029552022508234
- iter  1: compile=OK | correct=True | score=0.8114127546528617
- iter  2: compile=FAIL | correct=False | score=N/A
- iter  3: compile=OK | correct=True | score=0.8005779704525748

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
