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

    uint N = NX * NY;

    // Branchless periodic wrap for ±1 offsets.
    uint im1 = (i == 0u)       ? (NX - 1u) : (i - 1u);
    uint ip1 = (i == NX - 1u)  ? 0u        : (i + 1u);
    uint jm1 = (j == 0u)       ? (NY - 1u) : (j - 1u);
    uint jp1 = (j == NY - 1u)  ? 0u        : (j + 1u);

    // Row bases (in units of floats) for the three rows we need.
    uint row_j   = j   * NX;
    uint row_jm1 = jm1 * NX;
    uint row_jp1 = jp1 * NX;

    // Pull streaming: f_streamed[k] = f_in[k, (i-CX[k]) mod NX, (j-CY[k]) mod NY]
    // CX = {0, 1, 0,-1, 0, 1,-1,-1, 1}
    // CY = {0, 0, 1, 0,-1, 1, 1,-1,-1}
    // For k=0: src = (i, j)
    // For k=1: src = (i-1, j)        -> im1, j
    // For k=2: src = (i, j-1)        -> i, jm1
    // For k=3: src = (i+1, j)        -> ip1, j
    // For k=4: src = (i, j+1)        -> i, jp1
    // For k=5: src = (i-1, j-1)      -> im1, jm1
    // For k=6: src = (i+1, j-1)      -> ip1, jm1
    // For k=7: src = (i+1, j+1)      -> ip1, jp1
    // For k=8: src = (i-1, j+1)      -> im1, jp1

    float f0 = f_in[0u * N + row_j   + i  ];
    float f1 = f_in[1u * N + row_j   + im1];
    float f2 = f_in[2u * N + row_jm1 + i  ];
    float f3 = f_in[3u * N + row_j   + ip1];
    float f4 = f_in[4u * N + row_jp1 + i  ];
    float f5 = f_in[5u * N + row_jm1 + im1];
    float f6 = f_in[6u * N + row_jm1 + ip1];
    float f7 = f_in[7u * N + row_jp1 + ip1];
    float f8 = f_in[8u * N + row_jp1 + im1];

    // Moments.
    float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    float inv_rho = 1.0f / rho;
    // CX: 0,1,0,-1,0,1,-1,-1,1
    float mx = f1 - f3 + f5 - f6 - f7 + f8;
    // CY: 0,0,1,0,-1,1,1,-1,-1
    float my = f2 - f4 + f5 + f6 - f7 - f8;
    float ux = mx * inv_rho;
    float uy = my * inv_rho;

    float usq = ux * ux + uy * uy;
    float inv_tau = 1.0f / tau;
    float one_minus_inv_tau = 1.0f - inv_tau;

    const float W0 = 4.0f / 9.0f;
    const float W1 = 1.0f / 9.0f;
    const float W5 = 1.0f / 36.0f;

    float c1 = 1.0f - 1.5f * usq;

    // k=0: cu = 0
    float feq0 = W0 * rho * c1;
    // k=1: cu = ux
    float cu1 = ux;
    float feq1 = W1 * rho * (c1 + 3.0f * cu1 + 4.5f * cu1 * cu1);
    // k=2: cu = uy
    float cu2 = uy;
    float feq2 = W1 * rho * (c1 + 3.0f * cu2 + 4.5f * cu2 * cu2);
    // k=3: cu = -ux
    float cu3 = -ux;
    float feq3 = W1 * rho * (c1 + 3.0f * cu3 + 4.5f * cu3 * cu3);
    // k=4: cu = -uy
    float cu4 = -uy;
    float feq4 = W1 * rho * (c1 + 3.0f * cu4 + 4.5f * cu4 * cu4);
    // k=5: cu = ux + uy
    float cu5 = ux + uy;
    float feq5 = W5 * rho * (c1 + 3.0f * cu5 + 4.5f * cu5 * cu5);
    // k=6: cu = -ux + uy
    float cu6 = -ux + uy;
    float feq6 = W5 * rho * (c1 + 3.0f * cu6 + 4.5f * cu6 * cu6);
    // k=7: cu = -ux - uy
    float cu7 = -ux - uy;
    float feq7 = W5 * rho * (c1 + 3.0f * cu7 + 4.5f * cu7 * cu7);
    // k=8: cu = ux - uy
    float cu8 = ux - uy;
    float feq8 = W5 * rho * (c1 + 3.0f * cu8 + 4.5f * cu8 * cu8);

    uint idx = row_j + i;

    // f_out[k] = f[k] - inv_tau * (f[k] - feq[k])
    //         = (1 - inv_tau) * f[k] + inv_tau * feq[k]
    f_out[0u * N + idx] = one_minus_inv_tau * f0 + inv_tau * feq0;
    f_out[1u * N + idx] = one_minus_inv_tau * f1 + inv_tau * feq1;
    f_out[2u * N + idx] = one_minus_inv_tau * f2 + inv_tau * feq2;
    f_out[3u * N + idx] = one_minus_inv_tau * f3 + inv_tau * feq3;
    f_out[4u * N + idx] = one_minus_inv_tau * f4 + inv_tau * feq4;
    f_out[5u * N + idx] = one_minus_inv_tau * f5 + inv_tau * feq5;
    f_out[6u * N + idx] = one_minus_inv_tau * f6 + inv_tau * feq6;
    f_out[7u * N + idx] = one_minus_inv_tau * f7 + inv_tau * feq7;
    f_out[8u * N + idx] = one_minus_inv_tau * f8 + inv_tau * feq8;
}