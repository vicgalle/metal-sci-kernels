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
    y[i] = fma(a, x[i], y[i]);
}
```

Result of previous attempt:
                1M: correct, 0.09 ms, 144.1 GB/s (72.0% of 200 GB/s)
               16M: correct, 1.26 ms, 159.3 GB/s (79.7% of 200 GB/s)
               64M: correct, 4.32 ms, 186.5 GB/s (93.3% of 200 GB/s)
  score (gmean of fraction): 0.8120

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

- iter  2: compile=OK | correct=True | score=0.7752943364081034
- iter  3: compile=OK | correct=True | score=0.7745231198096424
- iter  4: compile=OK | correct=True | score=0.8122927681627791
- iter  5: compile=OK | correct=True | score=0.7938061312557994
- iter  6: compile=OK | correct=True | score=0.8339538847522566
- iter  7: compile=OK | correct=True | score=0.8067559343679458
- iter  8: compile=OK | correct=True | score=0.8139286360874372
- iter  9: compile=OK | correct=True | score=0.8119549847076338

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
