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
                       uint tsize [[threads_per_threadgroup]]) {
    
    threadgroup float4 shared_pos[1024];

    // Maintain full SIMD group execution to prevent deadlocks at threadgroup_barrier
    bool valid = i < N;
    
    float3 ri = valid ? pos_in[i].xyz : float3(0.0f);
    float3 vi = valid ? vel_in[i].xyz : float3(0.0f);
    float3 a  = float3(0.0f);
    float eps2 = eps * eps;

    for (uint j_start = 0; j_start < N; j_start += tsize) {
        
        // Cooperatively load tile into threadgroup memory
        uint j = j_start + tid;
        if (j < N) {
            // Premultiplying mass by G here saves instructions in the deeply nested loop
            shared_pos[tid] = float4(pos_in[j].xyz, G * mass[j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        uint tile_end = min(tsize, N - j_start);
        uint k = 0;
        
        // 8-way unrolled loop to maximize ILP and instruction throughput
        for (; k + 7 < tile_end; k += 8) {
            #pragma unroll
            for (uint u = 0; u < 8; ++u) {
                float4 pj = shared_pos[k + u];
                float3 d  = pj.xyz - ri;
                float r2  = dot(d, d) + eps2;
                
                float inv_r = rsqrt(r2);
                
                // Reassociated math for a shallower dependency chain:
                // Evaluates (pj.w * inv_r) and (inv_r * inv_r) in parallel
                float m_ir = pj.w * inv_r;
                float ir2 = inv_r * inv_r;
                
                a += d * (m_ir * ir2);
            }
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

    // Only output if the thread belongs to a valid particle
    if (valid) {
        float3 v_new = vi + a * dt;
        float3 r_new = ri + v_new * dt;

        pos_out[i] = float4(r_new, 0.0f);
        vel_out[i] = float4(v_new, 0.0f);
    }
}