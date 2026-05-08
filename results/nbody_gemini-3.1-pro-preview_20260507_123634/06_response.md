To optimize the previous memory-bound approach, we apply **threadgroup memory tiling** (cooperative loading) combined with **instruction-level pipelining (ILP)**. 

1. **Optimization**: We cooperatively load `pos` and `mass` into threadgroup memory, packing them into a single `float4` to optimize shared memory banks and maximize cache coalescence. We limit threadgroup size to 256 (`[[max_total_threads_per_threadgroup(256)]]`) to guarantee 128 registers per thread, entirely avoiding register spilling. The inner arithmetic loop is unrolled by 8 and computes forces using Array-of-Structures (AOS) `dot(d, d)`, which maps to single-cycle native dot product instructions on Apple Silicon. Accumulations are split across four independent `float3` registers (`a0`-`a3`) to hide Read-After-Write (RAW) latency. Vectorized `fast::rsqrt(float4)` evaluates roots in parallel.
2. **Why it improves**: Using threadgroup memory eliminates redundant global memory loads. Native `dot()` and split accumulations significantly boost ALU throughput, and the 256-thread limit guarantees optimal register occupancy compared to unhinted kernels.

```metal
#include <metal_stdlib>
using namespace metal;

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
                       uint i [[thread_position_in_grid]],
                       uint ti [[thread_position_in_threadgroup]],
                       uint tg_size [[threads_per_threadgroup]])
{
    float3 ri = float3(0.0f);
    float3 vi = float3(0.0f);
    
    // Mask out-of-bounds reads but keep threads active for barriers
    if (i < N) {
        ri = pos_in[i].xyz;
        vi = vel_in[i].xyz;
    }
    
    float eps2 = eps * eps;
    
    // 4 independent accumulators to hide FMA data dependency latency
    float3 a0 = float3(0.0f);
    float3 a1 = float3(0.0f);
    float3 a2 = float3(0.0f);
    float3 a3 = float3(0.0f);
    
    // Safely sized since we hint max_total_threads_per_threadgroup(256)
    threadgroup float4 shared_data[256];
    
    for (uint t = 0; t < N; t += tg_size) {
        uint limit = min(tg_size, N - t);
        
        // Cooperative load: pack position and mass into a single float4
        if (ti < limit) {
            float4 p = pos_in[t + ti];
            p.w = mass[t + ti];
            shared_data[ti] = p;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        uint k = 0;
        
        // Unroll by 8 to maximize ILP and vectorize math
        for (; k + 7 < limit; k += 8) {
            float4 data0 = shared_data[k];
            float4 data1 = shared_data[k+1];
            float4 data2 = shared_data[k+2];
            float4 data3 = shared_data[k+3];
            
            float3 d0 = data0.xyz - ri;
            float3 d1 = data1.xyz - ri;
            float3 d2 = data2.xyz - ri;
            float3 d3 = data3.xyz - ri;
            
            // native dot(float3, float3) evaluates in 1 fast instruction on Apple Silicon
            float r2_0 = dot(d0, d0) + eps2;
            float r2_1 = dot(d1, d1) + eps2;
            float r2_2 = dot(d2, d2) + eps2;
            float r2_3 = dot(d3, d3) + eps2;
            
            float4 r2_v0 = float4(r2_0, r2_1, r2_2, r2_3);
            float4 inv_r_v0 = fast::rsqrt(r2_v0);
            float4 inv_r3_v0 = inv_r_v0 * inv_r_v0 * inv_r_v0;
            
            float4 m_v0 = float4(data0.w, data1.w, data2.w, data3.w);
            float4 f_v0 = m_v0 * inv_r3_v0;
            
            a0 += f_v0.x * d0;
            a1 += f_v0.y * d1;
            a2 += f_v0.z * d2;
            a3 += f_v0.w * d3;
            
            float4 data4 = shared_data[k+4];
            float4 data5 = shared_data[k+5];
            float4 data6 = shared_data[k+6];
            float4 data7 = shared_data[k+7];
            
            float3 d4 = data4.xyz - ri;
            float3 d5 = data5.xyz - ri;
            float3 d6 = data6.xyz - ri;
            float3 d7 = data7.xyz - ri;
            
            float r2_4 = dot(d4, d4) + eps2;
            float r2_5 = dot(d5, d5) + eps2;
            float r2_6 = dot(d6, d6) + eps2;
            float r2_7 = dot(d7, d7) + eps2;
            
            float4 r2_v1 = float4(r2_4, r2_5, r2_6, r2_7);
            float4 inv_r_v1 = fast::rsqrt(r2_v1);
            float4 inv_r3_v1 = inv_r_v1 * inv_r_v1 * inv_r_v1;
            
            float4 m_v1 = float4(data4.w, data5.w, data6.w, data7.w);
            float4 f_v1 = m_v1 * inv_r3_v1;
            
            a0 += f_v1.x * d4;
            a1 += f_v1.y * d5;
            a2 += f_v1.z * d6;
            a3 += f_v1.w * d7;
        }
        
        // Remainder loop
        for (; k < limit; ++k) {
            float4 data = shared_data[k];
            float3 d = data.xyz - ri;
            float r2 = dot(d, d) + eps2;
            float inv_r = fast::rsqrt(r2);
            float f = data.w * inv_r * inv_r * inv_r;
            a0 += f * d;
        }
        
        // Ensure tile reads are complete before next overwrite
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Guard final writes
    if (i < N) {
        float3 a = (a0 + a1 + a2 + a3) * G;
        float3 v_new = vi + a * dt;
        float3 r_new = ri + v_new * dt;
        
        pos_out[i] = float4(r_new, 0.0f);
        vel_out[i] = float4(v_new, 0.0f);
    }
}
```