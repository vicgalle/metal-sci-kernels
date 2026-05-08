Looking at the results, the incumbent uses float4 vectorized loads with a leader/follower pattern. The 1M case is still only 71% efficiency, suggesting latency is limiting small sizes. For larger sizes we're at 93.4% which is close to peak.

The key insight: the previous approach wastes 3 out of 4 threads on early returns for the vector path. While this reduces memory transactions, those idle threads still consume GPU resources. A cleaner approach is to use `float4` loads naturally aligned to the grid — dispatch is 1-per-element but we can reinterpret the entire buffer as float4 and have each thread handle index `i` in the scalar space, with the compiler seeing through to vector ops. 

Better: use `simdgroup` coalescing with `float4` where thread `i` always works, but we shift to float4 indexing — thread `i` loads `float4` at position `i` (treating the array as float4-typed with N/4 elements). But the signature requires one output per thread position...

The cleanest improvement: avoid the leader/follower waste entirely. Cast buffers to float4 pointers and use `i` directly as float4 index for the bulk, with scalar fallback for tail. This means threads 0..N/4-1 each do one float4 op (covering 4 floats), and threads N/4..N-1 handle tail scalars — but the host dispatches N threads total, so threads N/4..N-1 would double-count. We can't change dispatch.

Best approach: keep the float4 vectorization but reduce branch divergence and ensure maximum memory-level parallelism with `[[max_total_threads_per_threadgroup(1024)]]` and clean code the compiler can auto-vectorize well.

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