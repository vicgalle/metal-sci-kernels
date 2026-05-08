To maximize performance on Apple Silicon, we must lean into the architecture's strengths: a massive, fast L1/L2 cache that elegantly handles 2D stencil reads without requiring manual threadgroup memory orchestration.

### Optimizations
1. **RNG Short-Circuiting:** The most expensive operation per-site is the dual-round Murmur3 RNG hash. We unconditionally fetch the acceptance probability `pa`. If `pa >= 1.0f` (which is true for all energy-lowering or energy-neutral flips), `u < pa` is mathematically guaranteed to be true since the max value of `u` is `0.99999994f`. We cleanly bypass the RNG calculation entirely in these cases, yielding bit-identical results with significantly fewer instructions.
2. **Uniform Hash Hoisting:** The first round of the RNG hash depends only on `seed` and `step_idx`. We perform this once at the top of the active code path so the compiler seamlessly hoists it as a warp-uniform scalar operation.
3. **Pointer Math / Register Pressure:** We abandoned the slow threadgroup halo loading from the previous attempt. Instead, we use flat pointer offsets for row calculations (`row_curr`, `row_prev`, `row_next`), minimizing redundant integer multiplications and letting the hardware coalescer seamlessly handle the native byte loads.

```metal
#include <metal_stdlib>
using namespace metal;

kernel void ising_step(device       char  *spins    [[buffer(0)]],
                       device const float *p_accept [[buffer(1)]],
                       constant uint  &NX           [[buffer(2)]],
                       constant uint  &NY           [[buffer(3)]],
                       constant uint  &step_idx     [[buffer(4)]],
                       constant uint  &seed         [[buffer(5)]],
                       uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint j = gid.y;
    if (i >= NX || j >= NY) return;

    // Checkerboard sub-pass mask
    uint color = step_idx & 1u;
    if (((i + j) & 1u) != color) return;

    // Precompute the site-invariant portion of the RNG hash.
    // The compiler can hoist this as a scalar uniform operation across the active warp.
    uint base_x = seed + step_idx * 0x9E3779B9u;
    base_x = (base_x ^ (base_x >> 16)) * 0x85EBCA6Bu;
    base_x = (base_x ^ (base_x >> 13)) * 0xC2B2AE35u;
    base_x = base_x ^ (base_x >> 16);

    // Periodic neighbor bounds calculation (branchless ternary)
    uint im = (i != 0u) ? (i - 1u) : (NX - 1u);
    uint ip = (i + 1u != NX) ? (i + 1u) : 0u;
    uint jm = (j != 0u) ? (j - 1u) : (NY - 1u);
    uint jp = (j + 1u != NY) ? (j + 1u) : 0u;

    // Pre-calculate base row pointers to eliminate repeated multiplications
    device char* row_curr = spins + j  * NX;
    device char* row_prev = spins + jm * NX;
    device char* row_next = spins + jp * NX;

    // L1 cache absorbs the unmasked byte-loads natively
    int s  = (int)row_curr[i];
    int sl = (int)row_curr[im];
    int sr = (int)row_curr[ip];
    int sd = (int)row_prev[i];
    int su = (int)row_next[i];

    int h    = sl + sr + sd + su;
    int prod = s * h;
    uint idx = (uint)(prod + 4) >> 1u;
    
    float pa = p_accept[idx];

    // Short-circuit: if acceptance probability is >= 1.0 (e.g. energy lowers), 
    // we bypass the costly RNG calculation entirely since u < 1.0 is always true.
    if (pa < 1.0f) {
        uint site_idx = j * NX + i;
        uint x = base_x ^ site_idx;
        
        // Second mix32 round
        x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
        x = (x ^ (x >> 13)) * 0xC2B2AE35u;
        x = x ^ (x >> 16);

        float u = float(x >> 8) * (1.0f / 16777216.0f);
        if (u >= pa) return;
    }

    // Flip accepted
    row_curr[i] = (char)(-s);
}
```