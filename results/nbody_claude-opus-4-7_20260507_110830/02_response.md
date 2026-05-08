**Optimization:** Fix the compile error by moving `[[max_total_threads_per_threadgroup(N)]]` to the correct position (before `kernel`), and keep the threadgroup-tiled cooperative-load strategy. Each threadgroup loads TILE bodies into shared memory once, then all threads in the group reuse them — cutting global memory traffic by ~TILE×. I also unroll the inner loop and use `fma` plus `rsqrt(r2*r2*r2)` for fewer ops. TILE=128 matches Apple's SIMD width × 4, giving good occupancy without exhausting threadgroup memory (128×16B = 2KB).

**Why it should beat incumbent:** Incumbent makes N global loads of pos+mass per body (O(N²) global traffic). Tiled version reduces to O(N²/TILE) global loads. For N=2048 that's ~16× less DRAM traffic, turning the kernel compute-bound where Apple Silicon shines.

```metal
#include <metal_stdlib>
using namespace metal;

#define TILE 128u

[[max_total_threads_per_threadgroup(TILE)]]
kernel void nbody_step(device const float4 *pos_in  [[buffer(0)]],
                       device       float4 *pos_out [[buffer(1)]],
                       device const float4 *vel_in  [[buffer(2)]],
                       device       float4 *vel_out [[buffer(3)]],
                       device const float  *mass    [[buffer(4)]],
                       constant uint        &N      [[buffer(5)]],
                       constant float       &dt     [[buffer(6)]],
                       constant float       &eps    [[buffer(7)]],
                       constant float       &G      [[buffer(8)]],
                       uint i   [[thread_position_in_grid]],
                       uint lid [[thread_position_in_threadgroup]])
{
    threadgroup float4 sdata[TILE];   // .xyz = pos, .w = mass

    bool active = (i < N);
    float3 ri = active ? pos_in[i].xyz : float3(0.0f);
    float3 vi = active ? vel_in[i].xyz : float3(0.0f);
    float3 a  = float3(0.0f);
    float  eps2 = eps * eps;

    uint num_tiles = (N + TILE - 1u) / TILE;

    for (uint t = 0; t < num_tiles; ++t) {
        uint j_base = t * TILE;
        uint j_load = j_base + lid;

        // Cooperative load: one body per thread into shared memory.
        if (j_load < N) {
            float3 pj = pos_in[j_load].xyz;
            float  mj = mass[j_load];
            sdata[lid] = float4(pj, mj);
        } else {
            sdata[lid] = float4(0.0f);   // mass=0 contributes nothing
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint tile_count = min(TILE, N - j_base);

        if (tile_count == TILE) {
            // Fully populated tile: unroll aggressively.
            #pragma unroll(8)
            for (uint k = 0; k < TILE; ++k) {
                float4 pj = sdata[k];
                float3 d  = pj.xyz - ri;
                float  r2 = fma(d.x, d.x, fma(d.y, d.y, fma(d.z, d.z, eps2)));
                float  inv_r3 = rsqrt(r2 * r2 * r2);
                float  s = pj.w * inv_r3;
                a = fma(d, s, a);
            }
        } else {
            for (uint k = 0; k < tile_count; ++k) {
                float4 pj = sdata[k];
                float3 d  = pj.xyz - ri;
                float  r2 = fma(d.x, d.x, fma(d.y, d.y, fma(d.z, d.z, eps2)));
                float  inv_r3 = rsqrt(r2 * r2 * r2);
                float  s = pj.w * inv_r3;
                a = fma(d, s, a);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active) {
        a *= G;
        float3 v_new = vi + a * dt;
        float3 r_new = ri + v_new * dt;
        pos_out[i] = float4(r_new, 0.0f);
        vel_out[i] = float4(v_new, 0.0f);
    }
}
```