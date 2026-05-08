#include <metal_stdlib>
using namespace metal;

constant float TWO_PI = 6.28318530717958647692f;

inline uint log2_uint(uint n) {
    uint r = 0u;
    while ((1u << r) < n) ++r;
    return r;
}

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Multiply by -i: (a + bi)*(-i) = b - a*i
inline float2 mul_minus_i(float2 a) { return float2(a.y, -a.x); }

// Stockham auto-sort FFT, radix-4 mixed with radix-2 final stage when logN is odd.
// `A`, `B`: ping-pong threadgroup buffers of size N (float2 each).
// `tw`: precomputed twiddle table of size N/2: tw[t] = exp(-2πi t / N), t in [0,N/2).
// Returns the buffer holding the final result.
inline threadgroup float2* line_fft_radix4(threadgroup float2 *A,
                                           threadgroup float2 *B,
                                           threadgroup float2 *tw,
                                           uint i, uint N, uint logN) {
    uint Nh = N >> 1u;

    // Cooperative twiddle precompute.
    if (i < Nh) {
        float ang = -TWO_PI * float(i) / float(N);
        float c, s;
        s = sincos(ang, c);
        tw[i] = float2(c, s);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float2 *cur = A;
    threadgroup float2 *nxt = B;

    uint Nq = N >> 2u;          // # of radix-4 butterflies per stage
    uint stages4 = logN >> 1u;  // # of radix-4 stages
    bool odd     = (logN & 1u) != 0u;

    // Radix-4 stages.
    // At stage s (0..stages4-1), Ns = 4^s = sub-FFT size before stage.
    // Each butterfly takes 4 inputs spaced by Ns within group of size 4*Ns.
    // We index butterfly id b in [0, Nq):
    //   group_id = b / Ns, j (within sub-FFT) = b % Ns.
    // Inputs: cur[group_id*Ns + j + r*Nh_sub] for r=0..3, where Nh_sub = N/4.
    // Wait: for Stockham radix-4, inputs for butterfly b are at
    //   in_r = b + r * (N/4), r=0..3
    // and outputs go to out_r = group_id*4*Ns + j + r*Ns.
    // Twiddle for input r is w^(r*j) where w = exp(-2πi/(4*Ns)).
    // In our base-N twiddle table, w^k = tw[k * (N/(4*Ns))] for k < 2*Ns.
    // For r=2,3 we need indices up to 3*(Ns-1) which can exceed Nh; use periodicity.
    for (uint s = 0u; s < stages4; ++s) {
        uint Ns   = 1u << (2u * s);          // 4^s
        uint mask = Ns - 1u;
        uint twStepBase = N >> (2u * s + 2u); // N / (4*Ns)

        if (i < Nq) {
            uint b  = i;
            uint j  = b & mask;
            uint g  = b >> (2u * s);          // group id
            uint outBase = (g << (2u * s + 2u)) + j; // g * 4*Ns + j

            // Load 4 inputs from cur.
            float2 x0 = cur[b + 0u * Nq];
            float2 x1 = cur[b + 1u * Nq];
            float2 x2 = cur[b + 2u * Nq];
            float2 x3 = cur[b + 3u * Nq];

            // Apply twiddles: x_r *= w^(r*j), w = exp(-2πi/(4*Ns)).
            // In base-N table: index = r*j*twStepBase, modulo N (table only stores
            // half period; use w^(k+N/2) = -w^k).
            // For stage 0 (Ns=1, j=0) all twiddles are 1; skip the multiplies.
            if (s != 0u) {
                uint idx1 = (j * twStepBase) & (Nh - 1u);
                uint idx2 = ((2u * j) * twStepBase) & (Nh - 1u);
                uint idx3 = ((3u * j) * twStepBase) & (Nh - 1u);
                // Determine sign flips from the high bit (whether index wrapped past N/2).
                uint raw1 = j * twStepBase;
                uint raw2 = (2u * j) * twStepBase;
                uint raw3 = (3u * j) * twStepBase;
                float2 w1 = tw[idx1]; if ((raw1 & Nh) != 0u) w1 = -w1;
                float2 w2 = tw[idx2]; if ((raw2 & Nh) != 0u) w2 = -w2;
                float2 w3 = tw[idx3]; if ((raw3 & Nh) != 0u) w3 = -w3;
                x1 = cmul(w1, x1);
                x2 = cmul(w2, x2);
                x3 = cmul(w3, x3);
            }

            // Radix-4 butterfly (DFT matrix for forward FFT, sign -i):
            //   y0 = x0 + x1 + x2 + x3
            //   y1 = x0 - i*x1 - x2 + i*x3
            //   y2 = x0 - x1 + x2 - x3
            //   y3 = x0 + i*x1 - x2 - i*x3
            float2 t02p = x0 + x2;
            float2 t02m = x0 - x2;
            float2 t13p = x1 + x3;
            float2 t13m_i = mul_minus_i(x1 - x3); // -i*(x1 - x3)

            float2 y0 = t02p + t13p;
            float2 y2 = t02p - t13p;
            float2 y1 = t02m + t13m_i;
            float2 y3 = t02m - t13m_i;

            nxt[outBase + 0u * Ns] = y0;
            nxt[outBase + 1u * Ns] = y1;
            nxt[outBase + 2u * Ns] = y2;
            nxt[outBase + 3u * Ns] = y3;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float2 *tmp = cur; cur = nxt; nxt = tmp;
    }

    // Final radix-2 stage if logN is odd.
    if (odd) {
        uint Ns_log = 2u * stages4;
        uint Ns     = 1u << Ns_log;     // sub-FFT size before final stage
        uint mask   = Ns - 1u;
        uint twStep = N >> (Ns_log + 1u); // N / (2*Ns)

        if (i < Nh) {
            uint j  = i & mask;
            uint k  = i >> Ns_log;
            uint in0 = i;
            uint in1 = i + Nh;

            float2 x0 = cur[in0];
            float2 x1 = cur[in1];

            float2 w = tw[j * twStep];
            float2 t = cmul(w, x1);

            uint outBase = (k << (Ns_log + 1u)) + j;
            nxt[outBase]      = x0 + t;
            nxt[outBase + Ns] = x0 - t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float2 *tmp = cur; cur = nxt; nxt = tmp;
    }

    return cur;
}

[[max_total_threads_per_threadgroup(1024)]]
kernel void fft3d_x(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint line = gid.y;
    uint k = line / N;
    uint j = line - k * N;

    threadgroup float2 A[1024];
    threadgroup float2 B[1024];
    threadgroup float2 tw[512];

    uint logN = log2_uint(N);
    uint base_addr = (k * N + j) * N;

    A[i] = in_data[base_addr + i];

    threadgroup float2 *res = line_fft_radix4(A, B, tw, i, N, logN);

    out_data[base_addr + i] = res[i];
}

[[max_total_threads_per_threadgroup(1024)]]
kernel void fft3d_y(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint j = gid.x;
    uint line = gid.y;
    uint k = line / N;
    uint i = line - k * N;

    threadgroup float2 A[1024];
    threadgroup float2 B[1024];
    threadgroup float2 tw[512];

    uint logN = log2_uint(N);
    uint plane_base = k * N * N + i;

    A[j] = in_data[plane_base + j * N];

    threadgroup float2 *res = line_fft_radix4(A, B, tw, j, N, logN);

    out_data[plane_base + j * N] = res[j];
}

[[max_total_threads_per_threadgroup(1024)]]
kernel void fft3d_z(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint k = gid.x;
    uint line = gid.y;
    uint j = line / N;
    uint i = line - j * N;

    threadgroup float2 A[1024];
    threadgroup float2 B[1024];
    threadgroup float2 tw[512];

    uint logN = log2_uint(N);
    uint NN = N * N;
    uint col_base = j * N + i;

    A[k] = in_data[k * NN + col_base];

    threadgroup float2 *res = line_fft_radix4(A, B, tw, k, N, logN);

    out_data[k * NN + col_base] = res[k];
}