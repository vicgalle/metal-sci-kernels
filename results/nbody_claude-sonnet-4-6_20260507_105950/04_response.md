Looking at the NaN failure: the previous tiled attempt had a bug where threads with `i >= N` still participated in barrier synchronization but read garbage data, and the tile_count calculation `N - tile_start` could underflow when `tile_start >= N`. Let me fix those issues and keep the threadgroup tiling optimization (which is the right approach for this memory-bound all-pairs kernel).

The key fix: use `min(N, tile_start + TILE_SIZE) - tile_start` safely, and ensure the guard is only at the write stage, not before barrier participation.

Additional optimizations: Use `rsqrt` with the standard `(r2)^1.5` form via `r2 * rsqrt(r2)` trick, and process 8 bodies per inner loop iteration to improve ILP.

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

    // All threads (including i >= N) must participate in barriers.
    // Load position/velocity only for valid threads.
    float3 ri   = float3(0.0f);
    float3 vi   = float3(0.0f);
    if (i < N) {
        ri = pos_in[i].xyz;
        vi = vel_in[i].xyz;
    }

    float3 a    = float3(0.0f);
    float  eps2 = eps * eps;

    uint num_tiles = (N + TILE - 1) / TILE;

    for (uint t = 0; t < num_tiles; ++t) {
        uint src = t * TILE + li;
        // Cooperative load: each thread loads one element into shared memory
        if (src < N) {
            tg_pos[li]  = pos_in[src];
            tg_mass[li] = mass[src];
        } else {
            // Pad with zero mass so they contribute nothing to acceleration
            tg_pos[li]  = float4(0.0f);
            tg_mass[li] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Number of valid bodies in this tile
        uint tile_end   = min(t * TILE + TILE, N);
        uint tile_start = t * TILE;
        uint tile_count = tile_end - tile_start;

        // Unrolled 8x inner loop for instruction-level parallelism
        uint jj = 0;
        for (; jj + 8 <= tile_count; jj += 8) {
            float3 d0 = tg_pos[jj+0].xyz - ri;
            float3 d1 = tg_pos[jj+1].xyz - ri;
            float3 d2 = tg_pos[jj+2].xyz - ri;
            float3 d3 = tg_pos[jj+3].xyz - ri;
            float3 d4 = tg_pos[jj+4].xyz - ri;
            float3 d5 = tg_pos[jj+5].xyz - ri;
            float3 d6 = tg_pos[jj+6].xyz - ri;
            float3 d7 = tg_pos[jj+7].xyz - ri;

            float r0sq = dot(d0, d0) + eps2;
            float r1sq = dot(d1, d1) + eps2;
            float r2sq = dot(d2, d2) + eps2;
            float r3sq = dot(d3, d3) + eps2;
            float r4sq = dot(d4, d4) + eps2;
            float r5sq = dot(d5, d5) + eps2;
            float r6sq = dot(d6, d6) + eps2;
            float r7sq = dot(d7, d7) + eps2;

            // inv_r3 = 1 / r^3 = rsqrt(r2) / r2
            float inv0 = rsqrt(r0sq); float inv0c = inv0 * inv0 * inv0;
            float inv1 = rsqrt(r1sq); float inv1c = inv1 * inv1 * inv1;
            float inv2 = rsqrt(r2sq); float inv2c = inv2 * inv2 * inv2;
            float inv3 = rsqrt(r3sq); float inv3c = inv3 * inv3 * inv3;
            float inv4 = rsqrt(r4sq); float inv4c = inv4 * inv4 * inv4;
            float inv5 = rsqrt(r5sq); float inv5c = inv5 * inv5 * inv5;
            float inv6 = rsqrt(r6sq); float inv6c = inv6 * inv6 * inv6;
            float inv7 = rsqrt(r7sq); float inv7c = inv7 * inv7 * inv7;

            a += d0 * (tg_mass[jj+0] * inv0c);
            a += d1 * (tg_mass[jj+1] * inv1c);
            a += d2 * (tg_mass[jj+2] * inv2c);
            a += d3 * (tg_mass[jj+3] * inv3c);
            a += d4 * (tg_mass[jj+4] * inv4c);
            a += d5 * (tg_mass[jj+5] * inv5c);
            a += d6 * (tg_mass[jj+6] * inv6c);
            a += d7 * (tg_mass[jj+7] * inv7c);
        }
        // Scalar remainder
        for (; jj < tile_count; ++jj) {
            float3 d  = tg_pos[jj].xyz - ri;
            float  r2 = dot(d, d) + eps2;
            float  inv = rsqrt(r2);
            float  invc = inv * inv * inv;
            a += d * (tg_mass[jj] * invc);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (i >= N) return;

    // Apply G once
    a *= G;

    float3 v_new = vi + a * dt;
    float3 r_new = ri + v_new * dt;

    pos_out[i] = float4(r_new, pos_in[i].w);
    vel_out[i] = float4(v_new, vel_in[i].w);
}
```