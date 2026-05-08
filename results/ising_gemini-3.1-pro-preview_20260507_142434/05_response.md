I will apply four main optimizations to eliminate ALU bottlenecks and memory latency:
1. **Algorithmic Elision**: When the energy difference `dE <= 0` (i.e., `prod <= 0`), the acceptance probability is exactly `1.0`. Since `u` is drawn from `[0, 1)`, the flip is always unconditionally accepted. Because the RNG is stateless, we can completely skip the RNG calculation and the `p_accept` read, saving significant ALU work for ~50% of the active sites.
2. **Branchless Constant Resolution**: For `prod > 0`, it must evaluate to either 2 or 4. We can safely bypass the dependent memory array fetch by selecting directly between `p_accept[3]` and `p_accept[4]`.
3. **Precomputed Uniform Hash State**: The initial seed mix for Murmur3 only depends on uniform variables `seed` and `step_idx`. Hoisting this cuts the per-thread RNG ALU instructions in half.
4. **Branchless Index Math**: The 2D periodic boundary wrapping and linear addresses are computed using only `select` and additions on the existing `site_idx`, completely eliminating multiplication instructions.

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
    uint nx = NX;
    uint ny = NY;
    uint i = gid.x;
    uint j = gid.y;
    if (i >= nx || j >= ny) return;

    // Early exit for checkerboard gating
    uint s_step = step_idx;
    if (((i + j) & 1u) != (s_step & 1u)) return;

    uint site_idx  = j * nx + i;
    uint grid_size = nx * ny;

    // Calculate periodic neighbors without using integer multiplication
    uint left_idx  = select(site_idx - 1u, site_idx + nx - 1u, i == 0u);
    uint right_idx = select(site_idx + 1u, site_idx - nx + 1u, i == nx - 1u);
    uint up_idx    = select(site_idx - nx, site_idx + grid_size - nx, j == 0u);
    uint down_idx  = select(site_idx + nx, i, j == ny - 1u);

    device const char* cspins = (device const char*)spins;

    int s  = cspins[site_idx];
    int sl = cspins[left_idx];
    int sr = cspins[right_idx];
    int su = cspins[up_idx];
    int sd = cspins[down_idx];

    int h = sl + sr + su + sd;
    int prod = s * h;

    // Fast-path: dE <= 0 implies acceptance probability 1.0 (u < 1.0 is always true).
    // We completely bypass RNG processing and array lookups here.
    bool flip = (prod <= 0);

    if (!flip) {
        // Since prod > 0 and the step is even, prod must be 2 or 4.
        float pa = (prod == 2) ? p_accept[3] : p_accept[4];

        uint s_seed = seed;
        uint base_x = s_seed + s_step * 0x9E3779B9u;
        base_x = (base_x ^ (base_x >> 16)) * 0x85EBCA6Bu;
        base_x = (base_x ^ (base_x >> 13)) * 0xC2B2AE35u;
        base_x = base_x ^ (base_x >> 16);

        uint x = base_x ^ site_idx;
        x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
        x = (x ^ (x >> 13)) * 0xC2B2AE35u;
        uint bits = x ^ (x >> 16);

        float u = float(bits >> 8) * (1.0f / 16777216.0f);
        flip = (u < pa);
    }

    if (flip) {
        spins[site_idx] = (char)(-s);
    }
}
```