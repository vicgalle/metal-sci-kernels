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
kernel void saxpy(device const float *x [[buffer(0)]],
                  device float       *y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) {
    if (i < N) {
        y[i] = fma(a, x[i], y[i]);
    }
}
```

Result of previous attempt:
                1M: correct, 0.09 ms, 144.3 GB/s (72.1% of 200 GB/s)
               16M: correct, 1.13 ms, 177.9 GB/s (89.0% of 200 GB/s)
               64M: correct, 4.39 ms, 183.5 GB/s (91.8% of 200 GB/s)
  score (gmean of fraction): 0.8381

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
                1M: correct, 0.04 ms, 337.2 GB/s (168.6% of 200 GB/s)
               16M: correct, 1.25 ms, 160.8 GB/s (80.4% of 200 GB/s)
               64M: correct, 4.32 ms, 186.3 GB/s (93.2% of 200 GB/s)
  score (gmean of fraction): 1.0808

## History

- iter  0: compile=OK | correct=True | score=1.0808190848898642
- iter  1: compile=OK | correct=True | score=0.8381313436870074
- iter  2: compile=FAIL | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
