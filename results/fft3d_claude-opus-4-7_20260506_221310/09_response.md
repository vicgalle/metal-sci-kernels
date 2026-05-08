**Optimization**: Switch to Stockham auto-sort radix-2 FFT (no bit-reversal), with cooperative twiddle precomputation cached in threadgroup memory, and use vectorized stride-1 loads/stores. Key changes vs incumbent:
1. Stockham eliminates the bit-reversal scatter (better load coalescing).
2. Twiddles `tw[t] = exp(-2πi t/N)` precomputed once per line into TG memory; butterflies just index into it (no per-stage `sin/cos`).
3. Each thread does one radix-2 butterfly per stage, but using ping-pong buffers avoids the in-place hazard from incumbent's read-then-write pattern.
4. Reduced barrier overhead — only one barrier per stage, same as incumbent, but each stage does less arithmetic (no sincos).

The previous radix-4 attempt regressed; sticking with simpler, well-optimized radix-2 Stockham with cached twiddles should be the cleanest win for memory-bound regimes (especially N=128).

```metal
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

// Stockham auto-sort radix-2 FFT. One TG of N threads cooperatively
// transforms a length-N line stored in `A`, using `B` as a ping-pong
// scratch and `tw` as a precomputed twiddle table of size N/2.
// Returns pointer to the buffer holding the final result.
inline threadgroup float2* line_fft_stockham(threadgroup float2 *A,
                                             threadgroup float2 *B,
                                             threadgroup float2 *tw,
                                             uint i, uint N, uint logN) {
    uint Nh = N >> 1u;

    // Cooperative twiddle precompute: tw[t] = exp(-2*pi*i * t / N), t in [0, N/2).
    if (i < Nh) {
        float ang = -TWO_PI * float(i) / float(N);
        float c, s;
        s = sincos(ang, c);
        tw[i] = float2(c, s);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float2 *cur = A;
    threadgroup float2 *nxt = B;

    // Stockham radix-2: at stage s (1..logN), Ns = 2^(s-1) is sub-FFT size
    // before this stage. Each butterfly takes (cur[k*Ns + j], cur[k*Ns + j + N/2])
    // -> (nxt[k*2*Ns + j], nxt[k*2*Ns + j + Ns]) with twiddle w^j, w = exp(-2pi i / (2Ns)).
    // Twiddle index in tw[] (which is base N): step = N / (2*Ns) = N >> s.
    for (uint s = 1u; s <= logN; ++s) {
        if (i < Nh) {
            uint Ns_log = s - 1u;
            uint Ns     = 1u << Ns_log;             // sub-FFT size before stage
            uint mask   = Ns - 1u;
            uint j      = i & mask;                 // index within sub-FFT
            uint k      = i >> Ns_log;              // sub-FFT id

            uint in0 = k * Ns + j;
            uint in1 = in0 + Nh;

            float2 x0 = cur[in0];
            float2 x1 = cur[in1];

            uint twStep = N >> s;                   // N / (2*Ns)
            float2 w = tw[j * twStep];
            float2 t = cmul(w, x1);

            uint outBase = (k << s) + j;            // k * (2*Ns) + j
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

    threadgroup float2 *res = line_fft_stockham(A, B, tw, i, N, logN);

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

    threadgroup float2 *res = line_fft_stockham(A, B, tw, j, N, logN);

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

    threadgroup float2 *res = line_fft_stockham(A, B, tw, k, N, logN);

    out_data[k * NN + col_base] = res[k];
}
```