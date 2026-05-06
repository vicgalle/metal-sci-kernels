#include <metal_stdlib>
using namespace metal;

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
                       uint tid [[thread_position_in_threadgroup]],
                       uint tsize [[threads_per_threadgroup]]) [[max_total_threads_per_threadgroup(1024)]] {
    
    threadgroup float4 shared_pos[1024];

    // Do not return early; keep SIMD-groups intact for the barrier to prevent deadlocks.
    bool valid = i < N;
    
    float3 ri = valid ? pos_in[i].xyz : float3(0.0f);
    float3 vi = valid ? vel_in[i].xyz : float3(0.0f);
    float3 a  = float3(0.0f);
    float eps2 = eps * eps;

    for (uint j_start = 0; j_start < N; j_start += tsize) {
        
        uint j = j_start + tid;
        if (j < N) {
            // Premultiplying mass by G directly on load saves instructions later
            shared_pos[tid] = float4(pos_in[j].xyz, G * mass[j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        uint tile_end = min(tsize, N - j_start);
        uint k = 0;
        
        // 8x unrolled loop, meticulously grouped to expose massive ILP
        for (; k + 7 < tile_end; k += 8) {
            float4 pj0 = shared_pos[k];
            float4 pj1 = shared_pos[k+1];
            float4 pj2 = shared_pos[k+2];
            float4 pj3 = shared_pos[k+3];
            float4 pj4 = shared_pos[k+4];
            float4 pj5 = shared_pos[k+5];
            float4 pj6 = shared_pos[k+6];
            float4 pj7 = shared_pos[k+7];
            
            float3 d0 = pj0.xyz - ri;
            float3 d1 = pj1.xyz - ri;
            float3 d2 = pj2.xyz - ri;
            float3 d3 = pj3.xyz - ri;
            float3 d4 = pj4.xyz - ri;
            float3 d5 = pj5.xyz - ri;
            float3 d6 = pj6.xyz - ri;
            float3 d7 = pj7.xyz - ri;

            float r2_0 = dot(d0, d0) + eps2;
            float r2_1 = dot(d1, d1) + eps2;
            float r2_2 = dot(d2, d2) + eps2;
            float r2_3 = dot(d3, d3) + eps2;
            float r2_4 = dot(d4, d4) + eps2;
            float r2_5 = dot(d5, d5) + eps2;
            float r2_6 = dot(d6, d6) + eps2;
            float r2_7 = dot(d7, d7) + eps2;

            float inv_r_0 = rsqrt(r2_0);
            float inv_r_1 = rsqrt(r2_1);
            float inv_r_2 = rsqrt(r2_2);
            float inv_r_3 = rsqrt(r2_3);
            float inv_r_4 = rsqrt(r2_4);
            float inv_r_5 = rsqrt(r2_5);
            float inv_r_6 = rsqrt(r2_6);
            float inv_r_7 = rsqrt(r2_7);

            float m_ir_0 = pj0.w * inv_r_0;
            float m_ir_1 = pj1.w * inv_r_1;
            float m_ir_2 = pj2.w * inv_r_2;
            float m_ir_3 = pj3.w * inv_r_3;
            float m_ir_4 = pj4.w * inv_r_4;
            float m_ir_5 = pj5.w * inv_r_5;
            float m_ir_6 = pj6.w * inv_r_6;
            float m_ir_7 = pj7.w * inv_r_7;

            float ir2_0 = inv_r_0 * inv_r_0;
            float ir2_1 = inv_r_1 * inv_r_1;
            float ir2_2 = inv_r_2 * inv_r_2;
            float ir2_3 = inv_r_3 * inv_r_3;
            float ir2_4 = inv_r_4 * inv_r_4;
            float ir2_5 = inv_r_5 * inv_r_5;
            float ir2_6 = inv_r_6 * inv_r_6;
            float ir2_7 = inv_r_7 * inv_r_7;

            float m_inv_r3_0 = m_ir_0 * ir2_0;
            float m_inv_r3_1 = m_ir_1 * ir2_1;
            float m_inv_r3_2 = m_ir_2 * ir2_2;
            float m_inv_r3_3 = m_ir_3 * ir2_3;
            float m_inv_r3_4 = m_ir_4 * ir2_4;
            float m_inv_r3_5 = m_ir_5 * ir2_5;
            float m_inv_r3_6 = m_ir_6 * ir2_6;
            float m_inv_r3_7 = m_ir_7 * ir2_7;

            a += d0 * m_inv_r3_0;
            a += d1 * m_inv_r3_1;
            a += d2 * m_inv_r3_2;
            a += d3 * m_inv_r3_3;
            a += d4 * m_inv_r3_4;
            a += d5 * m_inv_r3_5;
            a += d6 * m_inv_r3_6;
            a += d7 * m_inv_r3_7;
        }
        
        // Handle tail loop
        for (; k < tile_end; ++k) {
            float4 pj = shared_pos[k];
            float3 d  = pj.xyz - ri;
            float r2  = dot(d, d) + eps2;
            
            float inv_r = rsqrt(r2);
            float m_ir = pj.w * inv_r;
            float ir2 = inv_r * inv_r;
            
            a += d * (m_ir * ir2);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid) {
        float3 v_new = vi + a * dt;
        float3 r_new = ri + v_new * dt;

        pos_out[i] = float4(r_new, 0.0f);
        vel_out[i] = float4(v_new, 0.0f);
    }
}