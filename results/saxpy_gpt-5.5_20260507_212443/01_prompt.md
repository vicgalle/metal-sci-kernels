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

## Baseline: naive seed kernel

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

Measured baseline (seed):
                1M: correct, 0.09 ms, 144.4 GB/s (72.2% of 200 GB/s)
               16M: correct, 1.29 ms, 156.3 GB/s (78.1% of 200 GB/s)
               64M: correct, 4.39 ms, 183.5 GB/s (91.8% of 200 GB/s)
  score (gmean of fraction): 0.8030

## Your task

Write an improved Metal kernel that produces correct results AND runs
faster than the seed across all problem sizes. The fitness score is
the geometric mean of `achieved / ceiling` across sizes; score 0 if
any size fails correctness.

Output ONE fenced ```metal``` code block containing the kernel(s).
Preserve the kernel name(s) and buffer indices exactly.
