#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(1024)]]
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
    
    // Mask out-of-bounds reads safely while keeping threads active for barriers
    if (i < N) {
        ri = pos_in[i].xyz;
        vi = vel_in[i].xyz;
    }
    
    float eps2 = eps * eps;
    
    // 8 independent accumulators to fully hide FMA latency
    float3 a0 = float3(0.0f);
    float3 a1 = float3(0.0f);
    float3 a2 = float3(0.0f);
    float3 a3 = float3(0.0f);
    float3 a4 = float3(0.0f);
    float3 a5 = float3(0.0f);
    float3 a6 = float3(0.0f);
    float3 a7 = float3(0.0f);
    
    // Single packed threadgroup array for position (xyz) and mass (w)
    threadgroup float4 local_pos[1024] __attribute__((aligned(16)));
    
    for (uint t = 0; t < N; t += tg_size) {
        uint limit = min(tg_size, N - t);
        
        // Cooperative load: inject mass into the .w component of position
        if (ti < limit) {
            float4 p = pos_in[t + ti];
            p.w = mass[t + ti];
            local_pos[ti] = p;
        }
        
        // Ensure tile is fully loaded before computation begins
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        uint k = 0;
        // Unroll by 8 to maximize ILP and vectorize fast::rsqrt
        for (; k + 7 < limit; k += 8) {
            float4 p0 = local_pos[k];
            float4 p1 = local_pos[k+1];
            float4 p2 = local_pos[k+2];
            float4 p3 = local_pos[k+3];
            float4 p4 = local_pos[k+4];
            float4 p5 = local_pos[k+5];
            float4 p6 = local_pos[k+6];
            float4 p7 = local_pos[k+7];
            
            float3 d0 = p0.xyz - ri;
            float3 d1 = p1.xyz - ri;
            float3 d2 = p2.xyz - ri;
            float3 d3 = p3.xyz - ri;
            float3 d4 = p4.xyz - ri;
            float3 d5 = p5.xyz - ri;
            float3 d6 = p6.xyz - ri;
            float3 d7 = p7.xyz - ri;
            
            float4 r2_03 = float4(dot(d0, d0), dot(d1, d1), dot(d2, d2), dot(d3, d3)) + eps2;
            float4 r2_47 = float4(dot(d4, d4), dot(d5, d5), dot(d6, d6), dot(d7, d7)) + eps2;
            
            float4 inv_r_03 = fast::rsqrt(r2_03);
            float4 inv_r_47 = fast::rsqrt(r2_47);
            
            float4 inv_r3_03 = inv_r_03 * inv_r_03 * inv_r_03;
            float4 inv_r3_47 = inv_r_47 * inv_r_47 * inv_r_47;
            
            float4 f_03 = float4(p0.w, p1.w, p2.w, p3.w) * inv_r3_03;
            float4 f_47 = float4(p4.w, p5.w, p6.w, p7.w) * inv_r3_47;
            
            // Fully independent FMA chains
            a0 += f_03.x * d0;
            a1 += f_03.y * d1;
            a2 += f_03.z * d2;
            a3 += f_03.w * d3;
            
            a4 += f_47.x * d4;
            a5 += f_47.y * d5;
            a6 += f_47.z * d6;
            a7 += f_47.w * d7;
        }
        
        // Remainder loop
        for (; k < limit; ++k) {
            float4 p = local_pos[k];
            float3 d = p.xyz - ri;
            float r2 = dot(d, d) + eps2;
            float inv_r = fast::rsqrt(r2);
            float f = p.w * (inv_r * inv_r * inv_r);
            a0 += f * d;
        }
        
        // Ensure all computations are done before overwriting the next tile memory
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Guard the final write to memory
    if (i < N) {
        // Tree reduction of accumulators for better precision and speed
        float3 a = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
        a *= G;
        
        float3 v_new = vi + a * dt;
        float3 r_new = ri + v_new * dt;
        
        pos_out[i] = float4(r_new, 0.0f);
        vel_out[i] = float4(v_new, 0.0f);
    }
}