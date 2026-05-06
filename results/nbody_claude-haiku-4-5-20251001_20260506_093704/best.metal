// Naive seed kernel for all-pairs N-body gravity, one leapfrog step.
//
// Reads pos_in, vel_in, mass; writes pos_out, vel_out. Each thread updates
// one body i:
//   for j in 0..N: a += G m_j (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)
//   v_new = v_in + a * dt
//   x_new = x_in + v_new * dt
//
// Buffer layout:
//   buffer 0: const float4* pos_in   (N bodies; .xyz = position, .w unused)
//   buffer 1: device float4* pos_out
//   buffer 2: const float4* vel_in   (.xyz = velocity, .w unused)
//   buffer 3: device float4* vel_out
//   buffer 4: const float*  mass     (length N)
//   buffer 5: const uint&   N
//   buffer 6: const float&  dt
//   buffer 7: const float&  eps      (softening)
//   buffer 8: const float&  G

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
                       uint i [[thread_position_in_grid]]) {
    if (i >= N) return;

    float3 ri = pos_in[i].xyz;
    float3 vi = vel_in[i].xyz;
    float3 a  = float3(0.0f);

    float eps2 = eps * eps;
    for (uint j = 0; j < N; ++j) {
        float3 rj = pos_in[j].xyz;
        float  mj = mass[j];
        float3 d  = rj - ri;
        float  r2 = dot(d, d) + eps2;
        float  inv_r3 = rsqrt(r2 * r2 * r2);
        a += G * mj * d * inv_r3;
    }

    float3 v_new = vi + a * dt;
    float3 r_new = ri + v_new * dt;

    pos_out[i] = float4(r_new, 0.0f);
    vel_out[i] = float4(v_new, 0.0f);
}
