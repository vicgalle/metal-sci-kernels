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
                       uint lane_id [[thread_index_in_simdgroup]]) {
    
    if (i >= N) return;
    
    float3 ri = pos_in[i].xyz;
    float3 vi = vel_in[i].xyz;
    float3 a  = float3(0.0f);
    float eps2 = eps * eps;

    // Use simd_broadcast to share positions within the SIMD group.
    // This completely removes threadgroup memory overhead and synchronization barriers.
    for (uint j_start = 0; j_start < N; j_start += 32) {
        uint j = j_start + lane_id;
        
        // Load position and premultiply mass by G. Pad with 0s if out of bounds.
        float4 pj_local = float4(0.0f);
        if (j < N) {
            pj_local = float4(pos_in[j].xyz, G * mass[j]);
        }
        
        // 8-way manual unroll of the 32 SIMD broadcast lanes
        #pragma unroll
        for (uint k = 0; k < 32; k += 8) {
            float4 pj0 = simd_broadcast(pj_local, k);
            float4 pj1 = simd_broadcast(pj_local, k + 1);
            float4 pj2 = simd_broadcast(pj_local, k + 2);
            float4 pj3 = simd_broadcast(pj_local, k + 3);
            float4 pj4 = simd_broadcast(pj_local, k + 4);
            float4 pj5 = simd_broadcast(pj_local, k + 5);
            float4 pj6 = simd_broadcast(pj_local, k + 6);
            float4 pj7 = simd_broadcast(pj_local, k + 7);
            
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
            
            float ir_0 = rsqrt(r2_0);
            float ir_1 = rsqrt(r2_1);
            float ir_2 = rsqrt(r2_2);
            float ir_3 = rsqrt(r2_3);
            float ir_4 = rsqrt(r2_4);
            float ir_5 = rsqrt(r2_5);
            float ir_6 = rsqrt(r2_6);
            float ir_7 = rsqrt(r2_7);
            
            // Reassociate scalar math to minimize dependency chain length.
            // (pj.w * ir) and (ir * ir) evaluate in parallel.
            float s_0 = (pj0.w * ir_0) * (ir_0 * ir_0);
            float s_1 = (pj1.w * ir_1) * (ir_1 * ir_1);
            float s_2 = (pj2.w * ir_2) * (ir_2 * ir_2);
            float s_3 = (pj3.w * ir_3) * (ir_3 * ir_3);
            float s_4 = (pj4.w * ir_4) * (ir_4 * ir_4);
            float s_5 = (pj5.w * ir_5) * (ir_5 * ir_5);
            float s_6 = (pj6.w * ir_6) * (ir_6 * ir_6);
            float s_7 = (pj7.w * ir_7) * (ir_7 * ir_7);
            
            a += s_0 * d0;
            a += s_1 * d1;
            a += s_2 * d2;
            a += s_3 * d3;
            a += s_4 * d4;
            a += s_5 * d5;
            a += s_6 * d6;
            a += s_7 * d7;
        }
    }

    float3 v_new = vi + a * dt;
    float3 r_new = ri + v_new * dt;

    pos_out[i] = float4(r_new, 0.0f);
    vel_out[i] = float4(v_new, 0.0f);
}