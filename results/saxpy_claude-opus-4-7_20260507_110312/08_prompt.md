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

[[max_total_threads_per_threadgroup(1024)]]
kernel void saxpy(device const float * __restrict__ x [[buffer(0)]],
                  device float       * __restrict__ y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) {
    if (i >= N) return;
    const float xi = x[i];
    const float yi = y[i];
    y[i] = fma(a, xi, yi);
}
```

Result of previous attempt:
                1M: correct, 0.09 ms, 144.7 GB/s (72.4% of 200 GB/s)
               16M: correct, 1.26 ms, 159.6 GB/s (79.8% of 200 GB/s)
               64M: correct, 4.43 ms, 181.8 GB/s (90.9% of 200 GB/s)
  score (gmean of fraction): 0.8068

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

kernel void saxpy(device const float * __restrict__ x [[buffer(0)]],
                  device float       * __restrict__ y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) {
    if (i >= N) return;
    y[i] = fma(a, x[i], y[i]);
}
```

Incumbent result:
                1M: correct, 0.09 ms, 144.5 GB/s (72.3% of 200 GB/s)
               16M: correct, 1.16 ms, 173.3 GB/s (86.7% of 200 GB/s)
               64M: correct, 4.35 ms, 185.2 GB/s (92.6% of 200 GB/s)
  score (gmean of fraction): 0.8340

## History

- iter  0: compile=OK | correct=True | score=0.8172791072341589
- iter  1: compile=FAIL | correct=False | score=N/A
- iter  2: compile=OK | correct=True | score=0.7752943364081034
- iter  3: compile=OK | correct=True | score=0.7745231198096424
- iter  4: compile=OK | correct=True | score=0.8122927681627791
- iter  5: compile=OK | correct=True | score=0.7938061312557994
- iter  6: compile=OK | correct=True | score=0.8339538847522566
- iter  7: compile=OK | correct=True | score=0.8067559343679458

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
