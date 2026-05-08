**Optimization:** I'll keep the proven correct structure (TILE=128 with cooperative load) but increase ILP by processing 4 bodies per inner-loop iteration with independent accumulator chains, and use a single `float4` shared-memory load per body. The previous TILE=256 attempt likely produced NaN because `min(TILE, N - j_base)` underflows when `j_base > N` — but with TILE=128 and N a multiple of 128 this is safe. I'll also avoid the underflow risk by clamping. The 4-way unroll exposes more independent FMAs to the scheduler, increasing arithmetic throughput vs. the serial dependency chain on `a` in the incumbent.

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

    // 4 independent accumulators to break the FMA dependency chain.
    float3 a0 = float3(0.0f);
    float3 a1 = float3(0.0f);
    float3 a2 = float3(0.0f);
    float3 a3 = float3(0.0f);

    float eps2 = eps * eps;

    uint num_tiles = (N + TILE - 1u) / TILE;

    for (uint t = 0; t < num_tiles; ++t) {
        uint j_base = t * TILE;
        uint j_load = j_base + lid;

        // Cooperative load.
        if (j_load < N) {
            float3 pj = pos_in[j_load].xyz;
            float  mj = mass[j_load];
            sdata[lid] = float4(pj, mj);
        } else {
            sdata[lid] = float4(0.0f);   // mass=0 contributes nothing
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Guard against underflow when j_base >= N (shouldn't happen but safe).
        uint remaining = (j_base < N) ? (N - j_base) : 0u;
        uint tile_count = min(TILE, remaining);

        if (tile_count == TILE) {
            // Fully populated tile: 4-way ILP.
            #pragma unroll(4)
            for (uint k = 0; k < TILE; k += 4) {
                float4 p0 = sdata[k + 0];
                float4 p1 = sdata[k + 1];
                float4 p2 = sdata[k + 2];
                float4 p3 = sdata[k + 3];

                float3 d0 = p0.xyz - ri;
                float3 d1 = p1.xyz - ri;
                float3 d2 = p2.xyz - ri;
                float3 d3 = p3.xyz - ri;

                float r2_0 = fma(d0.x, d0.x, fma(d0.y, d0.y, fma(d0.z, d0.z, eps2)));
                float r2_1 = fma(d1.x, d1.x, fma(d1.y, d1.y, fma(d1.z, d1.z, eps2)));
                float r2_2 = fma(d2.x, d2.x, fma(d2.y, d2.y, fma(d2.z, d2.z, eps2)));
                float r2_3 = fma(d3.x, d3.x, fma(d3.y, d3.y, fma(d3.z, d3.z, eps2)));

                float s0 = p0.w * rsqrt(r2_0 * r2_0 * r2_0);
                float s1 = p1.w * rsqrt(r2_1 * r2_1 * r2_1);
                float s2 = p2.w * rsqrt(r2_2 * r2_2 * r2_2);
                float s3 = p3.w * rsqrt(r2_3 * r2_3 * r2_3);

                a0 = fma(d0, s0, a0);
                a1 = fma(d1, s1, a1);
                a2 = fma(d2, s2, a2);
                a3 = fma(d3, s3, a3);
            }
        } else {
            for (uint k = 0; k < tile_count; ++k) {
                float4 pj = sdata[k];
                float3 d  = pj.xyz - ri;
                float  r2 = fma(d.x, d.x, fma(d.y, d.y, fma(d.z, d.z, eps2)));
                float  inv_r3 = rsqrt(r2 * r2 * r2);
                float  s = pj.w * inv_r3;
                a0 = fma(d, s, a0);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active) {
        float3 a = (a0 + a1) + (a2 + a3);
        a *= G;
        float3 v_new = vi + a * dt;
        float3 r_new = ri + v_new * dt;
        pos_out[i] = float4(r_new, 0.0f);
        vel_out[i] = float4(v_new, 0.0f);
    }
}
```