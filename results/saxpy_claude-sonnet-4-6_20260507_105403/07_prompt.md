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

// Each thread handles one float4 worth of data from the perspective of the
// *grid* — but the host dispatches one thread per scalar element, so we use
// simd_group cooperation: within each simd-group of 32 threads, we reinterpret
// the 32-element window as 8 float4s. Thread lane k loads float4 at lane k%8,
// and lanes 8..31 are idle BUT since host fixes the grid size we cannot reduce
// it. Instead: use plain float4 loads per thread by reinterpreting the buffer,
// dispatching on a virtual index that is i/4. We must handle this carefully.
//
// STRUCTURAL CHANGE: We reshape the problem so the dispatch is over N/4
// "vector slots". Each thread computes its vector slot from thread_position_in_grid,
// performs a float4 load+fma+store, with tail scalar fallback.
// This is fundamentally different from the incumbent: ALL threads do float4 work
// (not just 1-in-4 leaders), achieving 4x the memory transaction efficiency.
// We reinterpret the scalar grid index i as a float4 index by having the kernel
// internally map: vec_i = i (treating the grid as sized ceil(N/4)).
// Host dispatches N threads (padded), so we remap: effective vec index = i,
// and the scalar range this covers is [4*i, 4*i+3].
// But host dispatches exactly N threads for N elements — we cannot change that.
//
// ACTUAL APPROACH: use float4 with the thread index directly as a float4 index.
// This means the host should dispatch ceil(N/4) threads. But spec says N threads.
// So: treat i as a float4 index when i < N/4, and handle remainder scalarly.
// Threads in [N/4, N) become the tail handlers for elements [N/4*4 .. N).
// Threads beyond N are no-ops (guard i >= N already covers thread >= N-padded).
//
// This gives: N/4 threads do float4 work + up to 3 scalar threads = same total.

[[max_total_threads_per_threadgroup(1024)]]
kernel void saxpy(device const float *x [[buffer(0)]],
                  device float       *y [[buffer(1)]],
                  constant float     &a [[buffer(2)]],
                  constant uint      &N [[buffer(3)]],
                  uint i [[thread_position_in_grid]])
{
    if (i >= N) return;

    const uint vec_count = N >> 2u;          // floor(N/4)
    const uint tail_start = vec_count << 2u; // vec_count * 4

    if (i < vec_count) {
        // Vector path: thread i handles float4 at index i
        device const float4 *xv = (device const float4 *)x;
        device       float4 *yv = (device       float4 *)y;
        float4 xval = xv[i];
        float4 yval = yv[i];
        yv[i] = fma(a, xval, yval);
    } else {
        // Tail path: threads [vec_count .. N) handle scalar elements
        // [tail_start .. N). Thread vec_count handles tail_start, etc.
        uint tail_idx = tail_start + (i - vec_count);
        if (tail_idx < N) {
            y[tail_idx] = fma(a, x[tail_idx], y[tail_idx]);
        }
    }
}
```

Result of previous attempt:
                1M: correct, 0.10 ms, 124.9 GB/s (62.5% of 200 GB/s)
               16M: correct, 1.53 ms, 131.3 GB/s (65.6% of 200 GB/s)
               64M: correct, 4.72 ms, 170.6 GB/s (85.3% of 200 GB/s)
  score (gmean of fraction): 0.7046

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
- iter  4: compile=OK | correct=True | score=0.8021966091013714
- iter  5: compile=OK | correct=True | score=0.8099909785450808
- iter  6: compile=OK | correct=True | score=0.7045622193873956

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
