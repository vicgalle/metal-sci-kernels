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
    // Boundary check to ensure we don't access memory out of bounds
    // when the grid size is rounded up to the threadgroup size.
    if (i < N) {
        // Load the multiplier 'a' into a register once per thread.
        const float alpha = a;
        
        // Reading x[i] and y[i] into local registers allows the hardware
        // to issue these memory requests in parallel.
        const float xi = x[i];
        const float yi = y[i];
        
        // Perform the calculation y = a * x + y using the fused multiply-add 
        // instruction, which is typically a single-cycle operation on 
        // Apple Silicon GPUs and minimizes precision loss.
        y[i] = fma(alpha, xi, yi);
    }
}
```

Result of previous attempt:
                1M: correct, 0.09 ms, 144.3 GB/s (72.1% of 200 GB/s)
               16M: correct, 1.27 ms, 159.1 GB/s (79.6% of 200 GB/s)
               64M: correct, 4.45 ms, 180.8 GB/s (90.4% of 200 GB/s)
  score (gmean of fraction): 0.8035

## Current best (incumbent)

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

Incumbent result:
                1M: correct, 0.09 ms, 144.1 GB/s (72.1% of 200 GB/s)
               16M: correct, 1.19 ms, 169.7 GB/s (84.9% of 200 GB/s)
               64M: correct, 4.31 ms, 187.0 GB/s (93.5% of 200 GB/s)
  score (gmean of fraction): 0.8301

## History

- iter  2: compile=OK | correct=True | score=0.8184898108134617
- iter  3: compile=OK | correct=True | score=0.8301040861563588
- iter  4: compile=OK | correct=True | score=0.8111347130405069
- iter  5: compile=FAIL | correct=False | score=N/A
- iter  6: compile=FAIL | correct=False | score=N/A
- iter  7: compile=OK | correct=True | score=0.826849845984189
- iter  8: compile=FAIL | correct=False | score=N/A
- iter  9: compile=OK | correct=True | score=0.8034976727357197

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
