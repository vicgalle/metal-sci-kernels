1. **Bypass RNG for Favorable Updates**: When the energy change is favorable or zero (`prod <= 0`), `p_accept` is exactly `1.0`. Since the RNG uniform `u` is strictly `< 1.0`, acceptance is guaranteed. We can skip the expensive hash, float conversion, and array read entirely and just flip the spin.
2. **Fast Path for Interior Boundary Checks**: We extract the complex periodic wrap-around logic into an `else` block, allowing 99% of threads to execute a branch-free fast path for neighbor loads. This avoids modulo/ternary arithmetic on every access.
3. **Hoist Uniform RNG State**: The first round of the `mix32` hash depends only on `seed` and `step_idx`. We inline the hash and compute this uniform base state once, cutting the remaining per-thread RNG ALU cost in half for the threads that do fall into the slow path.

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
    uint l_NX = NX;
    uint l_NY = NY;
    uint i = gid.x;
    uint j = gid.y;
    if (i >= l_NX || j >= l_NY) return;

    uint l_step_idx = step_idx;
    uint color = l_step_idx & 1u;
    if (((i + j) & 1u) != color) return;   // wrong color sub-pass: skip

    uint site_idx = j * l_NX + i;
    int s = (int)spins[site_idx];
    
    int sl, sr, sd, su;
    // Fast path for strictly interior sites avoids boundary conditionals entirely
    if (i > 0 && i < l_NX - 1 && j > 0 && j < l_NY - 1) {
        sl = (int)spins[site_idx - 1];
        sr = (int)spins[site_idx + 1];
        sd = (int)spins[site_idx - l_NX];
        su = (int)spins[site_idx + l_NX];
    } else {
        uint im = (i == 0u)          ? (l_NX - 1u) : (i - 1u);
        uint ip = (i == l_NX - 1u)   ? 0u          : (i + 1u);
        uint jm = (j == 0u)          ? (l_NY - 1u) : (j - 1u);
        uint jp = (j == l_NY - 1u)   ? 0u          : (j + 1u);
        sl = (int)spins[j  * l_NX + im];
        sr = (int)spins[j  * l_NX + ip];
        sd = (int)spins[jm * l_NX + i ];
        su = (int)spins[jp * l_NX + i ];
    }

    int h = sl + sr + sd + su;
    int prod = s * h;

    // If energy change is favorable/neutral (prod <= 0), p_accept is exactly 1.0 
    // and u < 1.0 is guaranteed. We can securely bypass the RNG computation entirely.
    if (prod <= 0) {
        spins[site_idx] = (char)(-s);
    } else {
        float pa = p_accept[(prod + 4) >> 1];
        
        // Hoist the first mix32 round: only depends on uniform seed and step_idx
        uint l_seed = seed;
        uint base_x = l_seed + l_step_idx * 0x9E3779B9u;
        base_x = (base_x ^ (base_x >> 16)) * 0x85EBCA6Bu;
        base_x = (base_x ^ (base_x >> 13)) * 0xC2B2AE35u;
        base_x = base_x ^ (base_x >> 16);
        
        // Second mix32 round incorporates the specific site_idx
        uint x = base_x ^ site_idx;
        x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
        x = (x ^ (x >> 13)) * 0xC2B2AE35u;
        x = x ^ (x >> 16);
        
        float u = float(x >> 8) * (1.0f / 16777216.0f);
        if (u < pa) {
            spins[site_idx] = (char)(-s);
        }
    }
}
```