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