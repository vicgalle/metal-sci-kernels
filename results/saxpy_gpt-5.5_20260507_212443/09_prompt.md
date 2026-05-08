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
    const float aa = a;

    // Fast path for SIMD-width blocks that are entirely in bounds.
    if ((i | 31u) < n) {
        const float xi = x[i];
        const float yi = y[i];
        y[i] = fma(aa, xi, yi);
        return;
    }

    // Tail / padded-grid path.
    if (i < n) {
        const float xi = x[i];
        const float yi = y[i];
        y[i] = fma(aa, xi, yi);
    }
}
```

Result of previous attempt:
                1M: correct, 0.09 ms, 143.5 GB/s (71.7% of 200 GB/s)
               16M: correct, 1.33 ms, 151.7 GB/s (75.8% of 200 GB/s)
               64M: correct, 4.66 ms, 172.9 GB/s (86.5% of 200 GB/s)
  score (gmean of fraction): 0.7777

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

- iter  1: compile=OK | correct=True | score=0.8114127546528617
- iter  2: compile=FAIL | correct=False | score=N/A
- iter  3: compile=OK | correct=True | score=0.8005779704525748
- iter  4: compile=FAIL | correct=False | score=N/A
- iter  5: compile=OK | correct=True | score=0.8017990245680746
- iter  6: compile=OK | correct=True | score=0.8102631635640112
- iter  7: compile=OK | correct=True | score=0.7991808286572587
- iter  8: compile=OK | correct=True | score=0.7776900565475625

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
