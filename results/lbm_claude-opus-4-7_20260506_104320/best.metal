#include <metal_stdlib>
using namespace metal;

kernel void lbm_step(device const float *f_in   [[buffer(0)]],
                     device       float *f_out  [[buffer(1)]],
                     constant uint        &NX   [[buffer(2)]],
                     constant uint        &NY   [[buffer(3)]],
                     constant float       &tau  [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint j = gid.y;
    if (i >= NX || j >= NY) return;

    uint nx = NX;
    uint ny = NY;
    uint N  = nx * ny;

    // Branchless periodic wrap for neighbor coordinates.
    uint ip1 = (i + 1u == nx) ? 0u : (i + 1u);
    uint im1 = (i == 0u) ? (nx - 1u) : (i - 1u);
    uint jp1 = (j + 1u == ny) ? 0u : (j + 1u);
    uint jm1 = (j == 0u) ? (ny - 1u) : (j - 1u);

    uint row    = j   * nx;
    uint rowU   = jm1 * nx;  // upstream for cy=+1 (source j-1)
    uint rowD   = jp1 * nx;  // upstream for cy=-1 (source j+1)

    // Pull-stream: f[k] = f_in[k, (i-cx) mod NX, (j-cy) mod NY]
    // k=0: (0,0)    -> (i, j)
    // k=1: (+1,0)   -> (i-1, j)
    // k=2: (0,+1)   -> (i, j-1)
    // k=3: (-1,0)   -> (i+1, j)
    // k=4: (0,-1)   -> (i, j+1)
    // k=5: (+1,+1)  -> (i-1, j-1)
    // k=6: (-1,+1)  -> (i+1, j-1)
    // k=7: (-1,-1)  -> (i+1, j+1)
    // k=8: (+1,-1)  -> (i-1, j+1)
    float f0 = f_in[0u * N + row  + i  ];
    float f1 = f_in[1u * N + row  + im1];
    float f2 = f_in[2u * N + rowU + i  ];
    float f3 = f_in[3u * N + row  + ip1];
    float f4 = f_in[4u * N + rowD + i  ];
    float f5 = f_in[5u * N + rowU + im1];
    float f6 = f_in[6u * N + rowU + ip1];
    float f7 = f_in[7u * N + rowD + ip1];
    float f8 = f_in[8u * N + rowD + im1];

    // Moments.
    float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    float jx  = (f1 - f3) + (f5 - f6) + (f8 - f7);
    float jy  = (f2 - f4) + (f5 + f6) - (f7 + f8);
    float inv_rho = 1.0f / rho;
    float ux = jx * inv_rho;
    float uy = jy * inv_rho;

    float usq = ux * ux + uy * uy;
    float inv_tau = 1.0f / tau;
    float omega = inv_tau;
    float one_minus_omega = 1.0f - omega;

    const float W0 = 4.0f / 9.0f;
    const float W1 = 1.0f / 9.0f;
    const float W5 = 1.0f / 36.0f;

    float k15 = 1.5f * usq;
    float c1  = 1.0f - k15;

    // Helper: feq_k = W_k * rho * (1 + 3 cu + 4.5 cu^2 - 1.5 usq)
    // Compute cu for each direction.
    float cu;
    float feq;
    uint idx = row + i;

    // k=0: cu = 0
    feq = W0 * rho * c1;
    f_out[0u * N + idx] = one_minus_omega * f0 + omega * feq;

    // k=1: cu = ux
    cu = ux;
    feq = W1 * rho * (c1 + 3.0f * cu + 4.5f * cu * cu);
    f_out[1u * N + idx] = one_minus_omega * f1 + omega * feq;

    // k=2: cu = uy
    cu = uy;
    feq = W1 * rho * (c1 + 3.0f * cu + 4.5f * cu * cu);
    f_out[2u * N + idx] = one_minus_omega * f2 + omega * feq;

    // k=3: cu = -ux
    cu = -ux;
    feq = W1 * rho * (c1 + 3.0f * cu + 4.5f * cu * cu);
    f_out[3u * N + idx] = one_minus_omega * f3 + omega * feq;

    // k=4: cu = -uy
    cu = -uy;
    feq = W1 * rho * (c1 + 3.0f * cu + 4.5f * cu * cu);
    f_out[4u * N + idx] = one_minus_omega * f4 + omega * feq;

    // k=5: cu = ux + uy
    cu = ux + uy;
    feq = W5 * rho * (c1 + 3.0f * cu + 4.5f * cu * cu);
    f_out[5u * N + idx] = one_minus_omega * f5 + omega * feq;

    // k=6: cu = -ux + uy
    cu = -ux + uy;
    feq = W5 * rho * (c1 + 3.0f * cu + 4.5f * cu * cu);
    f_out[6u * N + idx] = one_minus_omega * f6 + omega * feq;

    // k=7: cu = -ux - uy
    cu = -ux - uy;
    feq = W5 * rho * (c1 + 3.0f * cu + 4.5f * cu * cu);
    f_out[7u * N + idx] = one_minus_omega * f7 + omega * feq;

    // k=8: cu = ux - uy
    cu = ux - uy;
    feq = W5 * rho * (c1 + 3.0f * cu + 4.5f * cu * cu);
    f_out[8u * N + idx] = one_minus_omega * f8 + omega * feq;
}