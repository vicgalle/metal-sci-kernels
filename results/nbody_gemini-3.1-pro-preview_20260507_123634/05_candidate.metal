#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(64)]]
kernel void nbody_step(device const float4 *pos_in  [[buffer(0)]],
                       device       float4 *pos_out [[buffer(1)]],
                       device const float4 *vel_in  [[buffer(2)]],
                       device       float4 *vel_out [[buffer(3)]],
                       device const float  *mass    [[buffer(4)]],
                       constant uint        &N      [[buffer(5)]],
                       constant float       &dt     [[buffer(6)]],
                       constant float       &eps    [[buffer(7)]],
                       constant float       &G      [[buffer(8)]],
                       uint i [[thread_position_in_grid]])
{
    if (i >= N) return;
    
    float3 ri = pos_in[i].xyz;
    float3 vi = vel_in[i].xyz;
    float eps2 = eps * eps;
    
    // Split accumulators to hide FMA instruction latency
    float4 ax0 = 0.0f, ay0 = 0.0f, az0 = 0.0f;
    float4 ax1 = 0.0f, ay1 = 0.0f, az1 = 0.0f;
    
    // Cast mass buffer to float4 to vectorize loads
    device const float4* mass4 = (device const float4*)mass;
    
    uint j = 0;
    
    // Unroll by 8, processing 4 interactions per vector
    #pragma unroll(1)
    for (; j + 7 < N; j += 8) {
        float4 p0 = pos_in[j];
        float4 p1 = pos_in[j+1];
        float4 p2 = pos_in[j+2];
        float4 p3 = pos_in[j+3];
        float4 p4 = pos_in[j+4];
        float4 p5 = pos_in[j+5];
        float4 p6 = pos_in[j+6];
        float4 p7 = pos_in[j+7];
        
        float4 m0 = mass4[j >> 2];
        float4 m1 = mass4[(j >> 2) + 1];
        
        // Transpose spatial coordinates into vectorized lanes
        float4 dx0 = float4(p0.x, p1.x, p2.x, p3.x) - ri.x;
        float4 dy0 = float4(p0.y, p1.y, p2.y, p3.y) - ri.y;
        float4 dz0 = float4(p0.z, p1.z, p2.z, p3.z) - ri.z;
        
        float4 dx1 = float4(p4.x, p5.x, p6.x, p7.x) - ri.x;
        float4 dy1 = float4(p4.y, p5.y, p6.y, p7.y) - ri.y;
        float4 dz1 = float4(p4.z, p5.z, p6.z, p7.z) - ri.z;
        
        // Explicit nested fma for r2 calculation
        float4 r2_0 = fma(dx0, dx0, fma(dy0, dy0, fma(dz0, dz0, eps2)));
        float4 r2_1 = fma(dx1, dx1, fma(dy1, dy1, fma(dz1, dz1, eps2)));
        
        float4 inv_r_0 = fast::rsqrt(r2_0);
        float4 inv_r_1 = fast::rsqrt(r2_1);
        
        float4 inv_r3_0 = inv_r_0 * inv_r_0 * inv_r_0;
        float4 inv_r3_1 = inv_r_1 * inv_r_1 * inv_r_1;
        
        float4 f0 = m0 * inv_r3_0;
        float4 f1 = m1 * inv_r3_1;
        
        // Accumulate forces
        ax0 = fma(f0, dx0, ax0);
        ay0 = fma(f0, dy0, ay0);
        az0 = fma(f0, dz0, az0);
        
        ax1 = fma(f1, dx1, ax1);
        ay1 = fma(f1, dy1, ay1);
        az1 = fma(f1, dz1, az1);
    }
    
    // Reduce vector accumulators
    float3 a = float3(ax0.x + ax0.y + ax0.z + ax0.w + ax1.x + ax1.y + ax1.z + ax1.w,
                      ay0.x + ay0.y + ay0.z + ay0.w + ay1.x + ay1.y + ay1.z + ay1.w,
                      az0.x + az0.y + az0.z + az0.w + az1.x + az1.y + az1.z + az1.w);
                      
    // Remainder loop
    #pragma unroll(1)
    for (; j < N; ++j) {
        float3 p = pos_in[j].xyz;
        float3 d = p - ri;
        float r2 = fma(d.x, d.x, fma(d.y, d.y, fma(d.z, d.z, eps2)));
        float inv_r = fast::rsqrt(r2);
        float f = mass[j] * inv_r * inv_r * inv_r;
        a += f * d;
    }
    
    a *= G;
    float3 v_new = vi + a * dt;
    float3 r_new = ri + v_new * dt;
    
    pos_out[i] = float4(r_new, 0.0f);
    vel_out[i] = float4(v_new, 0.0f);
}