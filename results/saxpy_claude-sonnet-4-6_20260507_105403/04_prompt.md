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
    if (i >= N) return;
    y[i] = fma(a, x[i], y[i]);
}
```

Result of previous attempt:
                1M: correct, 0.09 ms, 144.4 GB/s (72.2% of 200 GB/s)
               16M: correct, 1.27 ms, 159.0 GB/s (79.5% of 200 GB/s)
               64M: correct, 4.31 ms, 186.8 GB/s (93.4% of 200 GB/s)
  score (gmean of fraction): 0.8124

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

kernel void saxpy(device const float *x [[buffer(0)]],
                  device float       *y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]])
{
    // Use float4 vectorized loads for groups of 4 aligned elements.
    // Threads whose index is a multiple of 4 load/store a float4 covering
    // indices [i, i+3], then return. The three "follower" threads (i%4 != 0)
    // that would have handled those same elements return immediately (no-op).
    // Threads in the tail (beyond the last aligned group of 4) fall back to
    // scalar. This reduces memory transaction count by ~4x and improves
    // memory-level parallelism for the bandwidth-bound small/medium cases.

    if (i >= N) return;

    // Check if this thread is the "leader" of a 4-wide vector group
    if ((i & 3u) == 0u) {
        uint remaining = N - i;
        if (remaining >= 4u) {
            // Vector path: load 4 floats at once
            device const float4 *xv = reinterpret_cast<device const float4*>(x + i);
            device float4       *yv = reinterpret_cast<device float4*>(y + i);
            float4 xval = *xv;
            float4 yval = *yv;
            *yv = fma(a, xval, yval);
            return;
        }
        // Fall through to scalar for tail < 4
    } else {
        // Non-leader threads: their element was handled by the leader above
        // if the leader used the vector path. Check if leader was in-bounds
        // and used vector path.
        uint leader = i & ~3u;
        if (leader < N && (N - leader) >= 4u) {
            // Already handled by leader thread
            return;
        }
        // Otherwise handle scalar (leader was in tail or out-of-bounds)
    }

    y[i] = fma(a, x[i], y[i]);
}
```

Incumbent result:
                1M: correct, 0.09 ms, 142.6 GB/s (71.3% of 200 GB/s)
               16M: correct, 1.17 ms, 172.2 GB/s (86.1% of 200 GB/s)
               64M: correct, 4.31 ms, 186.8 GB/s (93.4% of 200 GB/s)
  score (gmean of fraction): 0.8308

## History

- iter  0: compile=OK | correct=True | score=0.8118824633291214
- iter  1: compile=OK | correct=True | score=0.8118028901112135
- iter  2: compile=OK | correct=True | score=0.8307626037180771
- iter  3: compile=OK | correct=True | score=0.8123793637815585

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
