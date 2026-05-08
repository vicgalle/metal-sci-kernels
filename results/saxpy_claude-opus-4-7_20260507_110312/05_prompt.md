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

[[max_total_threads_per_threadgroup(256)]]
kernel void saxpy(device const float *x [[buffer(0)]],
                  device float       *y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) {
    if (i >= N) return;
    float xi = x[i];
    float yi = y[i];
    y[i] = fma(a, xi, yi);
}
```

Result of previous attempt:
                1M: correct, 0.09 ms, 145.0 GB/s (72.5% of 200 GB/s)
               16M: correct, 1.26 ms, 159.3 GB/s (79.6% of 200 GB/s)
               64M: correct, 4.34 ms, 185.6 GB/s (92.8% of 200 GB/s)
  score (gmean of fraction): 0.8123

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
                1M: correct, 0.09 ms, 144.4 GB/s (72.2% of 200 GB/s)
               16M: correct, 1.21 ms, 166.3 GB/s (83.2% of 200 GB/s)
               64M: correct, 4.43 ms, 181.8 GB/s (90.9% of 200 GB/s)
  score (gmean of fraction): 0.8173

## History

- iter  0: compile=OK | correct=True | score=0.8172791072341589
- iter  1: compile=FAIL | correct=False | score=N/A
- iter  2: compile=OK | correct=True | score=0.7752943364081034
- iter  3: compile=OK | correct=True | score=0.7745231198096424
- iter  4: compile=OK | correct=True | score=0.8122927681627791

## Stagnation notice

Your last 3 correct attempts all scored within 15% of
the incumbent without surpassing it. You are circling a local
optimum. STOP making incremental edits to the previous kernel.
Propose a STRUCTURALLY different approach — pick at least one of:

- Different memory layout (e.g. SoA <-> AoS; AA-pattern single-buffer
  in-place updates; pack scalars into a `float4` channel)
- Threadgroup-memory cooperative loads + `threadgroup_barrier` so a
  block of threads share one DRAM read across many compute steps
- SIMD-group cooperative reductions / shuffles via `simd_broadcast`,
  `simd_shuffle_*`, `simd_sum`, `simd_max`
- Multi-step / temporal kernel fusion: process several iterations
  per dispatch, holding state in threadgroup or register memory
- Persistent threadgroups (one threadgroup owns a tile across the
  entire run, eliminating per-step dispatch overhead)
- Mixed precision (fp16 inner loop with fp32 accumulation) where
  tolerance allows

A reworded version of the previous kernel will not break out of
this plateau.

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
