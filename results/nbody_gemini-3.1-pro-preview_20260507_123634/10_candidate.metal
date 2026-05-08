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
    
    // Mask out-of-bounds computation but keep threads active for threadgroup barriers
    if (i < N) {
        ri = pos_in[i].xyz;
        vi = vel_in[i].xyz;
    }
    
    float eps2 = eps * eps;
    
    // Independent float4 accumulators eliminate horizontal dependencies in the loop
    float4 ax_v0 = float4(0.0f);
    float4 ay_v0 = float4(0.0f);
    float4 az_v0 = float4(0.0f);
    
    float4 ax_v1 = float4(0.0f);
    float4 ay_v1 = float4(0.0f);
    float4 az_v1 = float4(0.0f);
    
    // Single shared array for pos (xyz) and mass (w), mapped to max threadgroup size
    threadgroup float4 local_pos[1024];
    
    for (uint t = 0; t < N; t += tg_size) {
        uint limit = min(tg_size, N - t);
        
        // Cooperative load: cleanly coalesced 16-byte (pos) and 4-byte (mass) reads
        if (ti < limit) {
            float4 p = pos_in[t + ti];
            p.w = mass[t + ti];
            local_pos[ti] = p;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        uint k = 0;
        // Unroll by 8 (two blocks of 4 bodies) to massively overlap fma and rsqrt latency
        for (; k + 7 < limit; k += 8) {
            float4 p0 = local_pos[k];
            float4 p1 = local_pos[k+1];
            float4 p2 = local_pos[k+2];
            float4 p3 = local_pos[k+3];
            float4 p4 = local_pos[k+4];
            float4 p5 = local_pos[k+5];
            float4 p6 = local_pos[k+6];
            float4 p7 = local_pos[k+7];
            
            // In-register AoS to SoA transpose (zero-cost logical swizzles)
            float4 px0 = float4(p0.x, p1.x, p2.x, p3.x);
            float4 py0 = float4(p0.y, p1.y, p2.y, p3.y);
            float4 pz0 = float4(p0.z, p1.z, p2.z, p3.z);
            float4 pm0 = float4(p0.w, p1.w, p2.w, p3.w);
            
            float4 px1 = float4(p4.x, p5.x, p6.x, p7.x);
            float4 py1 = float4(p4.y, p5.y, p6.y, p7.y);
            float4 pz1 = float4(p4.z, p5.z, p6.z, p7.z);
            float4 pm1 = float4(p4.w, p5.w, p6.w, p7.w);
            
            float4 dx0 = px0 - ri.x;
            float4 dy0 = py0 - ri.y;
            float4 dz0 = pz0 - ri.z;
            
            float4 dx1 = px1 - ri.x;
            float4 dy1 = py1 - ri.y;
            float4 dz1 = pz1 - ri.z;
            
            // 128-bit vector arithmetic matching hardware ALUs execution width
            float4 r2_0 = dx0*dx0 + dy0*dy0 + dz0*dz0 + eps2;
            float4 r2_1 = dx1*dx1 + dy1*dy1 + dz1*dz1 + eps2;
            
            float4 inv_r0 = fast::rsqrt(r2_0);
            float4 inv_r1 = fast::rsqrt(r2_1);
            
            // Minimize latency chain for 3rd power calculation
            float4 inv_r2_0 = inv_r0 * inv_r0;
            float4 inv_r2_1 = inv_r1 * inv_r1;
            
            float4 f0 = (pm0 * inv_r0) * inv_r2_0;
            float4 f1 = (pm1 * inv_r1) * inv_r2_1;
            
            ax_v0 = fma(f0, dx0, ax_v0);
            ay_v0 = fma(f0, dy0, ay_v0);
            az_v0 = fma(f0, dz0, az_v0);
            
            ax_v1 = fma(f1, dx1, ax_v1);
            ay_v1 = fma(f1, dy1, ay_v1);
            az_v1 = fma(f1, dz1, az_v1);
        }
        
        // Remainder tail loop
        for (; k < limit; ++k) {
            float4 p = local_pos[k];
            float3 d = p.xyz - ri;
            float r2 = dot(d, d) + eps2;
            float inv_r = fast::rsqrt(r2);
            float f = p.w * inv_r * inv_r * inv_r;
            
            ax_v0.x = fma(f, d.x, ax_v0.x);
            ay_v0.x = fma(f, d.y, ay_v0.x);
            az_v0.x = fma(f, d.z, az_v0.x);
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Only valid bodies finalize the horizontal sum and produce output
    if (i < N) {
        float ax = ax_v0.x + ax_v0.y + ax_v0.z + ax_v0.w + ax_v1.x + ax_v1.y + ax_v1.z + ax_v1.w;
        float ay = ay_v0.x + ay_v0.y + ay_v0.z + ay_v0.w + ay_v1.x + ay_v1.y + ay_v1.z + ay_v1.w;
        float az = az_v0.x + az_v0.y + az_v0.z + az_v0.w + az_v1.x + az_v1.y + az_v1.z + az_v1.w;
        
        float3 a = float3(ax, ay, az) * G;
        float3 v_new = vi + a * dt;
        float3 r_new = ri + v_new * dt;
        
        pos_out[i] = float4(r_new, 0.0f);
        vel_out[i] = float4(v_new, 0.0f);
    }
}