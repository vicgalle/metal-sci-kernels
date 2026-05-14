**Optimization:** The small sizes (N=32, 64) are dominated by dispatch overhead and not memory bandwidth — current perf is 44/55 GB/s while N=128 hits 186 GB/s. The biggest remaining win is removing redundant work in the hot path: (1) replace `compact3` decode with cheap masked-bit tests for the boundary check (we only need to know if any of x/y/z equals 0 or N-1, which is equivalent to checking whether `tid & XM` is 0 or equals the top x-bit pattern, etc.), (2) hoist the `m & XM/YM/ZM` once, (3) use `fma` for the final combine, and (4) add `[[max_total_threads_per_threadgroup(256)]]` so the compiler can allocate more registers per thread aggressively. The boundary test becomes pure bit-arithmetic with no `compact3` call at all — saving ~15 ops per thread, which matters at N=32 where the kernel is launch-bound.

```metal
#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(256)]]
kernel void morton_stencil(device const float *u_in   [[buffer(0)]],
                           device       float *u_out  [[buffer(1)]],
                           constant uint      &N      [[buffer(2)]],
                           constant uint      &logN   [[buffer(3)]],
                           constant float     &alpha  [[buffer(4)]],
                           uint tid [[thread_position_in_grid]])
{
    uint total = N * N * N;
    if (tid >= total) return;

    // Morton bit masks (full 32-bit; coords stay in [0,N) so high bits are 0).
    const uint XM = 0x09249249u;
    const uint YM = 0x12492492u;
    const uint ZM = 0x24924924u;

    uint m  = tid;
    uint mx = m & XM;
    uint my = m & YM;
    uint mz = m & ZM;

    // Boundary check without decoding: coord==0 iff its bit-plane is 0;
    // coord==N-1 iff its bit-plane equals the all-ones pattern truncated
    // to logN bits in that plane, i.e. XM & ((1u << (3*logN)) - 1u), etc.
    uint mask3 = (1u << (3u * logN)) - 1u;  // low 3*logN bits
    uint xm_all = XM & mask3;
    uint ym_all = YM & mask3;
    uint zm_all = ZM & mask3;

    if (mx == 0u || my == 0u || mz == 0u ||
        mx == xm_all || my == ym_all || mz == zm_all) {
        u_out[tid] = u_in[tid];
        return;
    }

    uint YZ = my | mz;
    uint XZ = mx | mz;
    uint XY = mx | my;

    // Neighbour Morton indices via masked carry/borrow on each bit-plane.
    uint m_xp = (((m | (YM | ZM)) + 1u) & XM) | YZ;
    uint m_xm = ((mx - 1u) & XM) | YZ;
    uint m_yp = (((m | (XM | ZM)) + 2u) & YM) | XZ;
    uint m_ym = ((my - 2u) & YM) | XZ;
    uint m_zp = (((m | (XM | YM)) + 4u) & ZM) | XY;
    uint m_zm = ((mz - 4u) & ZM) | XY;

    float c  = u_in[tid];
    float xp = u_in[m_xp];
    float xn = u_in[m_xm];
    float yp = u_in[m_yp];
    float yn = u_in[m_ym];
    float zp = u_in[m_zp];
    float zn = u_in[m_zm];

    float sum6 = (xp + xn) + (yp + yn) + (zp + zn);
    u_out[tid] = fma(alpha, sum6 - 6.0f * c, c);
}
```