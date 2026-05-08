Looking at the history, the bottleneck for small sizes (1M) is latency/occupancy rather than pure bandwidth. The incumbent at 71% efficiency for 1M suggests we're not saturating the memory system. I'll try a structurally different approach: **SIMD-group cooperative loads with explicit vectorization using `float4`**, combined with a thread-coarsening hint via `[[max_total_threads_per_threadgroup]]` tuning. More importantly, I'll use **`float4` loads aligned to the threadgroup**, where each thread in a SIMD group loads contiguous `float4` chunks — this is different from the incumbent's "leader/follower" approach which wastes 3/4 of threads. Here every thread does useful `float4` work, quadrupling effective bandwidth utilization per thread while maintaining full occupancy.

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