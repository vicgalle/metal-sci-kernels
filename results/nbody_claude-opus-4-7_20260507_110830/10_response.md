**Optimization & rationale (≤150 words):**

The incumbent uses TILE=128 (correct, score 0.055). Last attempt at TILE=256 failed correctness — likely due to `#pragma unroll(16)` interacting with reordering/precision. I'll keep TILE=128 (proven correct) but improve ILP by **unrolling the inner loop by 4 with explicit accumulation**, computing 4 interactions per iteration. This increases ILP without changing the math order significantly. I'll also hoist `eps2` and use `rsqrt` on `r2*r2*r2` (matching incumbent precision exactly to preserve correctness). Additionally, I add `[[max_total_threads_per_threadgroup(128)]]` and ensure the cooperative load reads `pos_in[j].xyz` and `mass[j]` cleanly. Computing 4 accumulations interleaved gives the scheduler more independent FMAs to hide rsqrt latency, which is the dominant cost on Apple GPUs. Math ordering stays bit-identical to incumbent (same fma chain), so correctness should hold while throughput improves.

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
    threadgroup float4 sdata[TILE];

    bool active = (i < N);
    float3 ri = active ? pos_in[i].xyz : float3(0.0f);
    float3 vi = active ? vel_in[i].xyz : float3(0.0f);
    float3 a  = float3(0.0f);
    float  eps2 = eps * eps;

    uint num_tiles = (N + TILE - 1u) / TILE;

    for (uint t = 0; t < num_tiles; ++t) {
        uint j_base = t * TILE;
        uint j_load = j_base + lid;

        if (j_load < N) {
            float3 pj = pos_in[j_load].xyz;
            float  mj = mass[j_load];
            sdata[lid] = float4(pj, mj);
        } else {
            sdata[lid] = float4(0.0f);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint tile_count = min(TILE, N - j_base);

        if (tile_count == TILE) {
            // Unroll by 4 for more ILP — independent FMAs hide rsqrt latency.
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

                float inv0 = rsqrt(r2_0 * r2_0 * r2_0);
                float inv1 = rsqrt(r2_1 * r2_1 * r2_1);
                float inv2 = rsqrt(r2_2 * r2_2 * r2_2);
                float inv3 = rsqrt(r2_3 * r2_3 * r2_3);

                float s0 = p0.w * inv0;
                float s1 = p1.w * inv1;
                float s2 = p2.w * inv2;
                float s3 = p3.w * inv3;

                a = fma(d0, s0, a);
                a = fma(d1, s1, a);
                a = fma(d2, s2, a);
                a = fma(d3, s3, a);
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