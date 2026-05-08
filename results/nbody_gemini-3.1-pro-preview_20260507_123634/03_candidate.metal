#include <metal_stdlib>
using namespace metal;

kernel void nbody_step(device const float4 *pos_in  [[buffer(0)]],
                       device       float4 *pos_out [[buffer(1)]],
                       device const float4 *vel_in  [[buffer(2)]],
                       device       float4 *vel_out [[buffer(3)]],
                       device const float  *mass    [[buffer(4)]],
                       constant uint        &N_ref  [[buffer(5)]],
                       constant float       &dt_ref [[buffer(6)]],
                       constant float       &eps_ref[[buffer(7)]],
                       constant float       &G_ref  [[buffer(8)]],
                       uint i [[thread_position_in_grid]],
                       uint ti [[thread_position_in_threadgroup]],
                       uint tg_size [[threads_per_threadgroup]]) [[max_total_threads_per_threadgroup(256)]]
{
    uint N = N_ref;
    float dt = dt_ref;
    float eps2 = eps_ref * eps_ref;
    float G = G_ref;
    
    // Mask out-of-bounds reads safely while keeping threads active for barriers
    float4 p_i = (i < N) ? pos_in[i] : float4(0.0f);
    float3 ri = p_i.xyz;
    float3 vi = (i < N) ? vel_in[i].xyz : float3(0.0f);
    
    // Split accumulators hide FMA latencies
    float3 a_0 = float3(0.0f);
    float3 a_1 = float3(0.0f);
    
    threadgroup float4 local_pos[256];
    threadgroup float  local_mass[256];
    
    for (uint t = 0; t < N; t += tg_size) {
        uint limit = min(tg_size, N - t);
        
        // Cooperative loading into threadgroup memory
        if (ti < limit) {
            local_pos[ti] = pos_in[t + ti];
            local_mass[ti] = mass[t + ti];
        }
        
        // Synchronize all active threads
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        uint k = 0;
        // Unroll by 8 to maximize ILP
        for (; k + 7 < limit; k += 8) {
            float3 d0 = local_pos[k].xyz - ri;
            float3 d1 = local_pos[k+1].xyz - ri;
            float3 d2 = local_pos[k+2].xyz - ri;
            float3 d3 = local_pos[k+3].xyz - ri;
            float3 d4 = local_pos[k+4].xyz - ri;
            float3 d5 = local_pos[k+5].xyz - ri;
            float3 d6 = local_pos[k+6].xyz - ri;
            float3 d7 = local_pos[k+7].xyz - ri;
            
            float r2_0 = dot(d0, d0) + eps2;
            float r2_1 = dot(d1, d1) + eps2;
            float r2_2 = dot(d2, d2) + eps2;
            float r2_3 = dot(d3, d3) + eps2;
            float r2_4 = dot(d4, d4) + eps2;
            float r2_5 = dot(d5, d5) + eps2;
            float r2_6 = dot(d6, d6) + eps2;
            float r2_7 = dot(d7, d7) + eps2;
            
            // Scalar fast rsqrt maps efficiently without pack/unpack overhead
            float inv_r_0 = fast::rsqrt(r2_0);
            float inv_r_1 = fast::rsqrt(r2_1);
            float inv_r_2 = fast::rsqrt(r2_2);
            float inv_r_3 = fast::rsqrt(r2_3);
            float inv_r_4 = fast::rsqrt(r2_4);
            float inv_r_5 = fast::rsqrt(r2_5);
            float inv_r_6 = fast::rsqrt(r2_6);
            float inv_r_7 = fast::rsqrt(r2_7);
            
            float f0 = local_mass[k]   * (inv_r_0 * inv_r_0 * inv_r_0);
            float f1 = local_mass[k+1] * (inv_r_1 * inv_r_1 * inv_r_1);
            float f2 = local_mass[k+2] * (inv_r_2 * inv_r_2 * inv_r_2);
            float f3 = local_mass[k+3] * (inv_r_3 * inv_r_3 * inv_r_3);
            float f4 = local_mass[k+4] * (inv_r_4 * inv_r_4 * inv_r_4);
            float f5 = local_mass[k+5] * (inv_r_5 * inv_r_5 * inv_r_5);
            float f6 = local_mass[k+6] * (inv_r_6 * inv_r_6 * inv_r_6);
            float f7 = local_mass[k+7] * (inv_r_7 * inv_r_7 * inv_r_7);
            
            a_0 += f0 * d0;
            a_1 += f1 * d1;
            a_0 += f2 * d2;
            a_1 += f3 * d3;
            a_0 += f4 * d4;
            a_1 += f5 * d5;
            a_0 += f6 * d6;
            a_1 += f7 * d7;
        }
        
        // Remainder loop
        for (; k < limit; ++k) {
            float3 d = local_pos[k].xyz - ri;
            float r2 = dot(d, d) + eps2;
            float inv_r = fast::rsqrt(r2);
            float f = local_mass[k] * (inv_r * inv_r * inv_r);
            a_0 += f * d;
        }
        
        // Synchronize before loading the next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Guard the final write to memory so only valid bodies produce output
    if (i < N) {
        float3 a = (a_0 + a_1) * G;
        float3 v_new = vi + a * dt;
        float3 r_new = ri + v_new * dt;
        
        pos_out[i] = float4(r_new, 0.0f);
        vel_out[i] = float4(v_new, 0.0f);
    }
}