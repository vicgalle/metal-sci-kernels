Looking at the failure: my tiled version was incorrect. The issue is likely numerical — the tiled version with 8x unrolling accumulates differently than the naive version. The tolerance is tight (0.002), and with 10 steps of leapfrog integration, small per-step errors compound.

Let me return to correctness-first approach: use the tiled cooperative load pattern (for bandwidth) but keep the inner loop simple (no aggressive unrolling that might reorder FP operations problematically). I'll use a tile size of 256 matching the threadgroup size, with a straightforward inner loop to match the reference accumulation order closely.

Key fix: iterate `jj` from 0 to `tile_count` in order, matching the sequential j-loop of the reference. The unrolled version accumulated in a different order causing FP divergence beyond tolerance.

```metal
#include <metal_stdlib>
using namespace metal;

constant uint TILE = 256;

[[max_total_threads_per_threadgroup(256)]]
kernel void nbody_step(device const float4 *pos_in  [[buffer(0)]],
                       device       float4 *pos_out [[buffer(1)]],
                       device const float4 *vel_in  [[buffer(2)]],
                       device       float4 *vel_out [[buffer(3)]],
                       device const float  *mass    [[buffer(4)]],
                       constant uint        &N      [[buffer(5)]],
                       constant float       &dt     [[buffer(6)]],
                       constant float       &eps    [[buffer(7)]],
                       constant float       &G      [[buffer(8)]],
                       uint i  [[thread_position_in_grid]],
                       uint li [[thread_position_in_threadgroup]])
{
    threadgroup float4 tg_pos[TILE];
    threadgroup float  tg_mass[TILE];

    float3 ri   = (i < N) ? pos_in[i].xyz : float3(0.0f);
    float3 vi   = (i < N) ? vel_in[i].xyz : float3(0.0f);

    float3 a    = float3(0.0f);
    float  eps2 = eps * eps;

    uint num_tiles = (N + TILE - 1) / TILE;

    for (uint t = 0; t < num_tiles; ++t) {
        uint src = t * TILE + li;

        if (src < N) {
            tg_pos[li]  = pos_in[src];
            tg_mass[li] = mass[src];
        } else {
            tg_pos[li]  = float4(1e18f, 0.0f, 0.0f, 0.0f);
            tg_mass[li] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint tile_start = t * TILE;
        uint tile_end   = min(tile_start + TILE, N);
        uint tile_count = tile_end - tile_start;

        // Simple sequential inner loop — matches reference FP order exactly.
        // The compiler will still vectorize/unroll as it sees fit.
        for (uint jj = 0; jj < tile_count; ++jj) {
            float3 d  = tg_pos[jj].xyz - ri;
            float  r2 = dot(d, d) + eps2;
            float  s  = rsqrt(r2 * r2 * r2);
            a += tg_mass[jj] * s * d;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (i >= N) return;

    a *= G;

    float3 v_new = vi + a * dt;
    float3 r_new = ri + v_new * dt;

    pos_out[i] = float4(r_new, pos_in[i].w);
    vel_out[i] = float4(v_new, vel_in[i].w);
}
```