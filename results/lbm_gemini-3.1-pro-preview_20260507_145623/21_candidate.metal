#include <metal_stdlib>
using namespace metal;

// Exploit symmetry, optimize BGK math, and hint max threadgroup size 
// to ensure sufficient threadgroups to fill GPU cores on small grids.
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
    uint nx = NX;
    uint ny = NY;
    
    if (i >= nx || j >= ny) return;

    uint N = nx * ny;
    uint idx = j * nx + i;

    // Fast 1D purely unsigned periodic boundary offset calculation.
    // Relies on 32-bit unsigned wrapping to natively avoid branching / modulo operators.
    uint im1 = (i > 0) ? -1u : (nx - 1);
    uint ip1 = (i < nx - 1) ? 1u : (1u - nx);
    uint jm1 = (j > 0) ? -nx : (nx * (ny - 1));
    uint jp1 = (j < ny - 1) ? nx : -(nx * (ny - 1));

    // Pull streamed values from neighbors
    float f0 = f_in[idx];
    float f1 = f_in[N + idx + im1];
    float f2 = f_in[2 * N + idx + jm1];
    float f3 = f_in[3 * N + idx + ip1];
    float f4 = f_in[4 * N + idx + jp1];
    float f5 = f_in[5 * N + idx + im1 + jm1];
    float f6 = f_in[6 * N + idx + ip1 + jm1];
    float f7 = f_in[7 * N + idx + ip1 + jp1];
    float f8 = f_in[8 * N + idx + im1 + jp1];

    // Compute macroscopic moments with optimized instruction grouping
    float f5p6 = f5 + f6;
    float f5m6 = f5 - f6;
    float f7p8 = f7 + f8;
    float f7m8 = f7 - f8;

    float rho = f0 + (f1 + f2) + (f3 + f4) + (f5p6 + f7p8);
    float ux  = (f1 - f3) + (f5m6 - f7m8);
    float uy  = (f2 - f4) + (f5p6 - f7p8);

    float inv_rho = 1.0f / rho;
    ux *= inv_rho;
    uy *= inv_rho;

    // Common terms for BGK collision
    float usq = ux * ux + uy * uy;
    float term0 = 1.0f - 1.5f * usq;

    float inv_tau_val = 1.0f / tau;
    float oma = 1.0f - inv_tau_val;
    float rt = rho * inv_tau_val;

    float w0_rt = (4.0f / 9.0f) * rt;
    float w1_rt = (1.0f / 9.0f) * rt;
    float w5_rt = (1.0f / 36.0f) * rt;

    // Execute collision. Exploit symmetry: opposite directions share the squared term.
    
    // k = 0
    f_out[idx] = f0 * oma + w0_rt * term0;

    // k = 1, 3
    float ux_sq = ux * ux;
    float term_x = term0 + 4.5f * ux_sq;
    float ux3 = 3.0f * ux;
    f_out[N + idx]     = f1 * oma + w1_rt * (term_x + ux3);
    f_out[3 * N + idx] = f3 * oma + w1_rt * (term_x - ux3);

    // k = 2, 4
    float uy_sq = uy * uy;
    float term_y = term0 + 4.5f * uy_sq;
    float uy3 = 3.0f * uy;
    f_out[2 * N + idx] = f2 * oma + w1_rt * (term_y + uy3);
    f_out[4 * N + idx] = f4 * oma + w1_rt * (term_y - uy3);

    // k = 5, 7
    float cu5 = ux + uy;
    float term_xy = term0 + 4.5f * (cu5 * cu5);
    float cu5_3 = 3.0f * cu5;
    f_out[5 * N + idx] = f5 * oma + w5_rt * (term_xy + cu5_3);
    f_out[7 * N + idx] = f7 * oma + w5_rt * (term_xy - cu5_3);

    // k = 6, 8
    float cu6 = -ux + uy;
    float term_xmy = term0 + 4.5f * (cu6 * cu6);
    float cu6_3 = 3.0f * cu6;
    f_out[6 * N + idx] = f6 * oma + w5_rt * (term_xmy + cu6_3);
    f_out[8 * N + idx] = f8 * oma + w5_rt * (term_xmy - cu6_3);
}