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
    
    if (i < N) {
        ri = pos_in[i].xyz;
        vi = vel_in[i].xyz;
    }
    
    float eps2 = eps * eps;
    float4 eps2_v = float4(eps2);
    
    // 4 independent sets of accumulators to hide loop-carried dependency latency
    float ax0 = 0.0f, ay0 = 0.0f, az0 = 0.0f;
    float ax1 = 0.0f, ay1 = 0.0f, az1 = 0.0f;
    float ax2 = 0.0f, ay2 = 0.0f, az2 = 0.0f;
    float ax3 = 0.0f, ay3 = 0.0f, az3 = 0.0f;
    
    // Structure-of-Arrays layout statically allocated to Metal's max threadgroup size (16 KB total)
    threadgroup float local_x[1024] __attribute__((aligned(16)));
    threadgroup float local_y[1024] __attribute__((aligned(16)));
    threadgroup float local_z[1024] __attribute__((aligned(16)));
    threadgroup float local_m[1024] __attribute__((aligned(16)));
    
    for (uint t = 0; t < N; t += tg_size) {
        uint limit = min(tg_size, N - t);
        
        // Cooperative load converting from AoS (global) to SoA (threadgroup)
        if (ti < limit) {
            float4 p = pos_in[t + ti];
            local_x[ti] = p.x;
            local_y[ti] = p.y;
            local_z[ti] = p.z;
            local_m[ti] = mass[t + ti];
        }
        
        // Wait for all tile data to be visible
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        uint k = 0;
        // Unroll by 16: processes 16 particles across 4 block iterations per loop step
        for (; k + 15 < limit; k += 16) {
            // Block 0
            float4 px0 = *(threadgroup float4*)(&local_x[k]);
            float4 py0 = *(threadgroup float4*)(&local_y[k]);
            float4 pz0 = *(threadgroup float4*)(&local_z[k]);
            float4 m0  = *(threadgroup float4*)(&local_m[k]);
            
            // Block 1
            float4 px1 = *(threadgroup float4*)(&local_x[k+4]);
            float4 py1 = *(threadgroup float4*)(&local_y[k+4]);
            float4 pz1 = *(threadgroup float4*)(&local_z[k+4]);
            float4 m1  = *(threadgroup float4*)(&local_m[k+4]);

            // Block 2
            float4 px2 = *(threadgroup float4*)(&local_x[k+8]);
            float4 py2 = *(threadgroup float4*)(&local_y[k+8]);
            float4 pz2 = *(threadgroup float4*)(&local_z[k+8]);
            float4 m2  = *(threadgroup float4*)(&local_m[k+8]);
            
            // Block 3
            float4 px3 = *(threadgroup float4*)(&local_x[k+12]);
            float4 py3 = *(threadgroup float4*)(&local_y[k+12]);
            float4 pz3 = *(threadgroup float4*)(&local_z[k+12]);
            float4 m3  = *(threadgroup float4*)(&local_m[k+12]);
            
            float4 dx0 = px0 - ri.x;
            float4 dy0 = py0 - ri.y;
            float4 dz0 = pz0 - ri.z;
            
            float4 dx1 = px1 - ri.x;
            float4 dy1 = py1 - ri.y;
            float4 dz1 = pz1 - ri.z;

            float4 dx2 = px2 - ri.x;
            float4 dy2 = py2 - ri.y;
            float4 dz2 = pz2 - ri.z;
            
            float4 dx3 = px3 - ri.x;
            float4 dy3 = py3 - ri.y;
            float4 dz3 = pz3 - ri.z;
            
            float4 r2_0 = fma(dz0, dz0, fma(dy0, dy0, fma(dx0, dx0, eps2_v)));
            float4 r2_1 = fma(dz1, dz1, fma(dy1, dy1, fma(dx1, dx1, eps2_v)));
            float4 r2_2 = fma(dz2, dz2, fma(dy2, dy2, fma(dx2, dx2, eps2_v)));
            float4 r2_3 = fma(dz3, dz3, fma(dy3, dy3, fma(dx3, dx3, eps2_v)));
            
            float4 inv_r_0 = fast::rsqrt(r2_0);
            float4 inv_r_1 = fast::rsqrt(r2_1);
            float4 inv_r_2 = fast::rsqrt(r2_2);
            float4 inv_r_3 = fast::rsqrt(r2_3);
            
            float4 inv_r3_0 = inv_r_0 * (inv_r_0 * inv_r_0);
            float4 inv_r3_1 = inv_r_1 * (inv_r_1 * inv_r_1);
            float4 inv_r3_2 = inv_r_2 * (inv_r_2 * inv_r_2);
            float4 inv_r3_3 = inv_r_3 * (inv_r_3 * inv_r_3);
            
            float4 f0 = m0 * inv_r3_0;
            float4 f1 = m1 * inv_r3_1;
            float4 f2 = m2 * inv_r3_2;
            float4 f3 = m3 * inv_r3_3;
            
            // Apple Silicon's 1-cycle dot matches mathematical component summation across 4 particles
            ax0 += dot(f0, dx0);
            ay0 += dot(f0, dy0);
            az0 += dot(f0, dz0);
            
            ax1 += dot(f1, dx1);
            ay1 += dot(f1, dy1);
            az1 += dot(f1, dz1);

            ax2 += dot(f2, dx2);
            ay2 += dot(f2, dy2);
            az2 += dot(f2, dz2);
            
            ax3 += dot(f3, dx3);
            ay3 += dot(f3, dy3);
            az3 += dot(f3, dz3);
        }
        
        // Remainder loop
        for (; k < limit; ++k) {
            float dx = local_x[k] - ri.x;
            float dy = local_y[k] - ri.y;
            float dz = local_z[k] - ri.z;
            
            float r2 = fma(dz, dz, fma(dy, dy, fma(dx, dx, eps2)));
            float inv_r = fast::rsqrt(r2);
            float f = local_m[k] * (inv_r * inv_r * inv_r);
            
            ax0 += f * dx;
            ay0 += f * dy;
            az0 += f * dz;
        }
        
        // Guard before overwriting tile arrays on the next iteration
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Combine and apply updates
    if (i < N) {
        float3 a = float3(ax0 + ax1 + ax2 + ax3,
                          ay0 + ay1 + ay2 + ay3,
                          az0 + az1 + az2 + az3) * G;
        
        float3 v_new = vi + a * dt;
        float3 r_new = ri + v_new * dt;
        
        pos_out[i] = float4(r_new, 0.0f);
        vel_out[i] = float4(v_new, 0.0f);
    }
}