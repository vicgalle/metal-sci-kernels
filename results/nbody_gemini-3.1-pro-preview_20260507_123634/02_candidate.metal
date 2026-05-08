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
                       uint i [[thread_position_in_grid]])
{
    uint N = N_ref;
    if (i >= N) return;
    
    float dt = dt_ref;
    float eps = eps_ref;
    float G = G_ref;
    
    float4 p_i = pos_in[i];
    float3 ri = p_i.xyz;
    float3 vi = vel_in[i].xyz;
    
    float3 a = float3(0.0f);
    float eps2 = eps * eps;
    
    uint j = 0;
    // Unroll by 8 to maximize Instruction-Level Parallelism (ILP)
    for (; j + 7 < N; j += 8) {
        float4 p0 = pos_in[j];
        float4 p1 = pos_in[j+1];
        float4 p2 = pos_in[j+2];
        float4 p3 = pos_in[j+3];
        float4 p4 = pos_in[j+4];
        float4 p5 = pos_in[j+5];
        float4 p6 = pos_in[j+6];
        float4 p7 = pos_in[j+7];
        
        // Coalesced loads for the scalar mass array
        float4 m_vec0 = *(device const float4*)(&mass[j]);
        float4 m_vec1 = *(device const float4*)(&mass[j+4]);
        
        float3 d0 = p0.xyz - ri;
        float3 d1 = p1.xyz - ri;
        float3 d2 = p2.xyz - ri;
        float3 d3 = p3.xyz - ri;
        float3 d4 = p4.xyz - ri;
        float3 d5 = p5.xyz - ri;
        float3 d6 = p6.xyz - ri;
        float3 d7 = p7.xyz - ri;
        
        float r2_0 = dot(d0, d0) + eps2;
        float r2_1 = dot(d1, d1) + eps2;
        float r2_2 = dot(d2, d2) + eps2;
        float r2_3 = dot(d3, d3) + eps2;
        float r2_4 = dot(d4, d4) + eps2;
        float r2_5 = dot(d5, d5) + eps2;
        float r2_6 = dot(d6, d6) + eps2;
        float r2_7 = dot(d7, d7) + eps2;
        
        // Pack into float4 vectors for fast SIMD reciprocal square root
        float4 r2_vec0 = float4(r2_0, r2_1, r2_2, r2_3);
        float4 r2_vec1 = float4(r2_4, r2_5, r2_6, r2_7);
        
        float4 inv_r_vec0 = fast::rsqrt(r2_vec0);
        float4 inv_r_vec1 = fast::rsqrt(r2_vec1);
        
        float4 inv_r3_vec0 = inv_r_vec0 * inv_r_vec0 * inv_r_vec0;
        float4 inv_r3_vec1 = inv_r_vec1 * inv_r_vec1 * inv_r_vec1;
        
        float4 f_vec0 = m_vec0 * inv_r3_vec0;
        float4 f_vec1 = m_vec1 * inv_r3_vec1;
        
        a = fma(f_vec0.x, d0, a);
        a = fma(f_vec0.y, d1, a);
        a = fma(f_vec0.z, d2, a);
        a = fma(f_vec0.w, d3, a);
        a = fma(f_vec1.x, d4, a);
        a = fma(f_vec1.y, d5, a);
        a = fma(f_vec1.z, d6, a);
        a = fma(f_vec1.w, d7, a);
    }
    
    // Process remaining elements if N is not a multiple of 8
    for (; j < N; ++j) {
        float4 p = pos_in[j];
        float m = mass[j];
        float3 d = p.xyz - ri;
        float r2 = dot(d, d) + eps2;
        float inv_r = fast::rsqrt(r2);
        float inv_r3 = inv_r * inv_r * inv_r;
        float f = m * inv_r3;
        a = fma(f, d, a);
    }
    
    // Apply the factored gravitational constant
    a *= G;
    
    float3 v_new = vi + a * dt;
    float3 r_new = ri + v_new * dt;
    
    pos_out[i] = float4(r_new, 0.0f);
    vel_out[i] = float4(v_new, 0.0f);
}