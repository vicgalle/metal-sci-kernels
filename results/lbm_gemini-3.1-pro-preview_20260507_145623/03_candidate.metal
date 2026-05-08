#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(256)]]
kernel void lbm_step(device const float *f_in   [[buffer(0)]],
                     device       float *f_out  [[buffer(1)]],
                     constant uint        &NX   [[buffer(2)]],
                     constant uint        &NY   [[buffer(3)]],
                     constant float       &tau  [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]])
{
    uint i = gid.x;
    uint j = gid.y;
    if (i >= NX || j >= NY) return;

    uint N = NX * NY;
    
    // Fast, branchless periodic boundary wrap
    uint im1 = (i == 0) ? NX - 1 : i - 1;
    uint ip1 = (i == NX - 1) ? 0 : i + 1;
    uint jm1 = (j == 0) ? NY - 1 : j - 1;
    uint jp1 = (j == NY - 1) ? 0 : j + 1;

    // Precalculate row offsets
    uint jNX   = j * NX;
    uint jm1NX = jm1 * NX;
    uint jp1NX = jp1 * NX;

    // 1. Pull streaming directly into explicit registers.
    // This absolutely guarantees no thread-local array spilling.
    float f0 = f_in[          jNX   + i];
    float f1 = f_in[1 * N + jNX   + im1];
    float f2 = f_in[2 * N + jm1NX + i];
    float f3 = f_in[3 * N + jNX   + ip1];
    float f4 = f_in[4 * N + jp1NX + i];
    float f5 = f_in[5 * N + jm1NX + im1];
    float f6 = f_in[6 * N + jm1NX + ip1];
    float f7 = f_in[7 * N + jp1NX + ip1];
    float f8 = f_in[8 * N + jp1NX + im1];

    // 2. Compute macroscopic moments
    float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    float ux  = f1 - f3 + f5 - f6 - f7 + f8;
    float uy  = f2 - f4 + f5 + f6 - f7 - f8;
    
    float inv_rho = 1.0f / rho;
    ux *= inv_rho;
    uy *= inv_rho;

    // 3. BGK collision step using exact explicitly unrolled math.
    float usq = ux * ux + uy * uy;
    float inv_tau = 1.0f / tau;
    float base_feq = 1.0f - 1.5f * usq;
    uint idx_out = jNX + i;

    // Precalculate weight factors
    float feq_rho_w0 = (4.0f /  9.0f) * rho;
    float feq_rho_w1 = (1.0f /  9.0f) * rho;
    float feq_rho_w2 = (1.0f / 36.0f) * rho;

    f_out[          idx_out] = f0 - inv_tau * (f0 - feq_rho_w0 * base_feq);

    float cu1 = ux;
    f_out[1 * N + idx_out] = f1 - inv_tau * (f1 - feq_rho_w1 * (base_feq + 3.0f * cu1 + 4.5f * cu1 * cu1));

    float cu2 = uy;
    f_out[2 * N + idx_out] = f2 - inv_tau * (f2 - feq_rho_w1 * (base_feq + 3.0f * cu2 + 4.5f * cu2 * cu2));

    float cu3 = -ux;
    f_out[3 * N + idx_out] = f3 - inv_tau * (f3 - feq_rho_w1 * (base_feq + 3.0f * cu3 + 4.5f * cu3 * cu3));

    float cu4 = -uy;
    f_out[4 * N + idx_out] = f4 - inv_tau * (f4 - feq_rho_w1 * (base_feq + 3.0f * cu4 + 4.5f * cu4 * cu4));

    float cu5 = ux + uy;
    f_out[5 * N + idx_out] = f5 - inv_tau * (f5 - feq_rho_w2 * (base_feq + 3.0f * cu5 + 4.5f * cu5 * cu5));

    float cu6 = -ux + uy;
    f_out[6 * N + idx_out] = f6 - inv_tau * (f6 - feq_rho_w2 * (base_feq + 3.0f * cu6 + 4.5f * cu6 * cu6));

    float cu7 = -ux - uy;
    f_out[7 * N + idx_out] = f7 - inv_tau * (f7 - feq_rho_w2 * (base_feq + 3.0f * cu7 + 4.5f * cu7 * cu7));

    float cu8 = ux - uy;
    f_out[8 * N + idx_out] = f8 - inv_tau * (f8 - feq_rho_w2 * (base_feq + 3.0f * cu8 + 4.5f * cu8 * cu8));
}