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
                  uint i [[thread_position_in_grid]])
{
    // Each thread handles exactly one element. We use float4 vectorized
    // loads for cache-line efficiency: leader threads (i % 4 == 0) issue
    // a 16-byte load/store covering 4 elements; follower threads (i % 4 != 0)
    // that fall within a complete vector group early-exit (their element was
    // handled by the leader). Tail elements use scalar. The key improvement
    // over the previous attempt: we hoist the constant `a` into a float4
    // splat and use fma directly, reducing register pressure and letting the
    // compiler schedule loads and FMAs optimally.

    if (i >= N) return;

    uint aligned_N = N & ~3u;  // largest multiple of 4 <= N

    if (i < aligned_N) {
        // All threads with i < aligned_N are in the vector region.
        // Leader threads (i % 4 == 0) do the float4 work.
        // Follower threads return immediately.
        if ((i & 3u) == 0u) {
            device const float4 *xv = reinterpret_cast<device const float4*>(x) + (i >> 2);
            device       float4 *yv = reinterpret_cast<device float4*>(y) + (i >> 2);
            float4 av = float4(a);
            float4 yval = *yv;
            float4 xval = *xv;
            *yv = fma(av, xval, yval);
        }
        // followers do nothing — their element was handled by leader
    } else {
        // Scalar tail for i in [aligned_N, N)
        y[i] = fma(a, x[i], y[i]);
    }
}
```

Result of previous attempt:
                1M: correct, 0.09 ms, 141.9 GB/s (71.0% of 200 GB/s)
               16M: correct, 1.13 ms, 177.6 GB/s (88.8% of 200 GB/s)
               64M: correct, 4.32 ms, 186.5 GB/s (93.3% of 200 GB/s)
  score (gmean of fraction): 0.8378

## History

- iter  1: compile=OK | correct=True | score=0.8118028901112135
- iter  2: compile=OK | correct=True | score=0.8307626037180771
- iter  3: compile=OK | correct=True | score=0.8123793637815585
- iter  4: compile=OK | correct=True | score=0.8021966091013714
- iter  5: compile=OK | correct=True | score=0.8099909785450808
- iter  6: compile=OK | correct=True | score=0.7045622193873956
- iter  7: compile=OK | correct=True | score=0.7933560315697554
- iter  8: compile=OK | correct=True | score=0.8377599615108304

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
