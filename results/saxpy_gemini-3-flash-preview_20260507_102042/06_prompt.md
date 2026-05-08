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

kernel void saxpy(device const float * restrict x [[buffer(0)]],
                  device float * restrict y [[buffer(1)]],
                  constant float &a [[buffer(2)]],
                  constant uint &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) {
    // Standard boundary check to handle dispatches padded to threadgroup size.
    if (i >= N) return;

    // Load constant 'a' once into a register.
    const float alpha = a;

    // Load x[i] and y[i] into registers to maximize Memory-Level Parallelism.
    // The restrict qualifier on the pointers tells the compiler that x and y
    // are independent, allowing it to issue these loads simultaneously.
    const float xi = x[i];
    const float yi = y[i];

    // Compute y = a * x + y using the fused multiply-add instruction.
    // This provides high throughput and precision in a single cycle.
    y[i] = fma(alpha, xi, yi);
}
```

Result of previous attempt:
  COMPILE FAILED: Error Domain=MTLLibraryErrorDomain Code=3 "program_source:4:49: error: expected ')'
kernel void saxpy(device const float * restrict x [[buffer(0)]],
                                                ^
program_source:4:18: note: to match this '('
kernel void saxpy(device const float * restrict x [[buffer(0)]],
                 ^
program_source:10:9: error: use of undeclared identifier 'i'
    if (i >= N) return;
        ^
program_source:10:14: error: use of undeclared identifier 'N'
    if (i >= N) return;
             ^
program_source:13:25: error: use of undeclared identifier 'a'
    const float alpha = a;
                        ^
program_source:18:22: error: use of undeclared identifier 'x'
    const float xi = x[i];
                     ^
program_source:18:24: error: use of undeclared identifier 'i'
    const float xi = x[i];
                       ^
program_source:19:22: error: use of undeclared identifier 'y'
    const float yi = y[i];
                     ^
program_source:19:24: error: use of undeclared identifier 'i'
    const float yi = y[i];
                       ^
program_source:23:5: error: use of undeclared identifier 'y'
    y[i] = fma(alpha, xi, yi);
    ^
program_source:23:7: error: use of undeclared identifier 'i'
    y[i] = fma(alpha, xi, yi);
      ^
" UserInfo={NSLocalizedDescription=program_source:4:49: error: expected ')'
kernel void saxpy(device const float * restrict x [[buffer(0)]],
                                                ^
program_source:4:18: note: to match this '('
kernel void saxpy(device const float * restrict x [[buffer(0)]],
                 ^
program_source:10:9: error: use of undeclared identifier 'i'
    if (i >= N) return;
        ^
program_source:10:14: error: use of undeclared identifier 'N'
    if (i >= N) return;
             ^
program_source:13:25: error: use of undeclared identifier 'a'
    const float alpha = a;
                        ^
program_source:18:22: error: use of undeclared identifier 'x'
    const float xi = x[i];
                     ^
program_source:18:24: error: use of undeclared identifier 'i'
    const float xi = x[i];
                       ^
program_source:19:22: error: use of undeclared identifier 'y'
    const float yi = y[i];
                     ^
program_source:19:24: error: use of undeclared identifier 'i'
    const float yi = y[i];
                       ^
program_source:23:5: error: use of undeclared identifier 'y'
    y[i] = fma(alpha, xi, yi);
    ^
program_source:23:7: error: use of undeclared identifier 'i'
    y[i] = fma(alpha, xi, yi);
      ^
}

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

- iter  0: compile=OK | correct=True | score=0.8135375071068955
- iter  1: compile=OK | correct=True | score=0.8051676453474895
- iter  2: compile=OK | correct=True | score=0.8184898108134617
- iter  3: compile=OK | correct=True | score=0.8301040861563588
- iter  4: compile=OK | correct=True | score=0.8111347130405069
- iter  5: compile=FAIL | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
