// Naive seed for 3D complex-to-complex forward FFT (fp32, power-of-two cube N).
//
// Three named kernels (one per axis) — the host calls them in order
// fft3d_x -> fft3d_y -> fft3d_z, ping-ponging between two buffers. Each
// kernel is one length-N 1D FFT done by one threadgroup of N threads in
// shared memory: cooperative load with bit-reversal permutation, then
// log2(N) Cooley-Tukey butterfly stages with sin/cos twiddles computed
// on the fly. Stride-1 reads/writes for x; strided N (or N*N) for y, z.
//
// Convention: forward, unnormalized. Y_k = sum_n X_n * exp(-2πi k n / N).
// Matches numpy.fft.fftn (norm="backward", default).
//
// Dispatch (host-provided, identical across all three kernels):
//   threadsPerGrid       = (N, N*N, 1)
//   threadsPerThreadgroup= (N, 1,  1)
// So gid.x ∈ [0,N) is the position along the FFT axis (= lid.x), and
// gid.y ∈ [0,N²) is the line index within the (N×N) plane orthogonal to
// the FFT axis.
//
// Storage: row-major float2[NZ][NY][NX] with NX=NY=NZ=N. Linear index of
// element (i,j,k) is ((k*N + j)*N + i). float2 = (real, imag).

#include <metal_stdlib>
using namespace metal;

constant float TWO_PI = 6.28318530717958647692f;

inline uint log2_uint(uint n) {
    uint r = 0u;
    while ((1u << r) < n) ++r;
    return r;
}

inline uint bit_reverse(uint x, uint logN) {
    uint r = 0u;
    for (uint b = 0u; b < logN; ++b) {
        r = (r << 1) | (x & 1u);
        x >>= 1u;
    }
    return r;
}

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// One TG of N threads cooperates on one length-N FFT in `buf`. The caller
// is responsible for loading `buf` with the bit-reversed input and
// writing back the result; this routine performs only the butterfly
// stages with a barrier between each.
inline void line_fft_butterflies(threadgroup float2 *buf,
                                 uint i, uint N, uint logN) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 1u; s <= logN; ++s) {
        uint m      = 1u << s;       // current butterfly span
        uint half_m = m >> 1u;
        if (i < (N >> 1u)) {
            uint group = i / half_m;
            uint kk    = i - group * half_m;
            uint base  = group * m;
            float angle = -TWO_PI * float(kk) / float(m);
            float c = cos(angle), s_ = sin(angle);
            float2 w = float2(c, s_);
            float2 t = cmul(w, buf[base + half_m + kk]);
            float2 u = buf[base + kk];
            buf[base + kk]          = u + t;
            buf[base + half_m + kk] = u - t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// FFT along x (stride 1). Line index gid.y decodes (j, k); element gid.x = i.
kernel void fft3d_x(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint line = gid.y;
    uint k = line / N;
    uint j = line - k * N;

    threadgroup float2 buf[1024];
    uint logN = log2_uint(N);

    if (i < N) {
        uint src = (k * N + j) * N + i;
        buf[bit_reverse(i, logN)] = in_data[src];
    }
    line_fft_butterflies(buf, i, N, logN);
    if (i < N) {
        uint dst = (k * N + j) * N + i;
        out_data[dst] = buf[i];
    }
}

// FFT along y (stride N). Line index gid.y decodes (i, k); element gid.x = j.
kernel void fft3d_y(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint j = gid.x;
    uint line = gid.y;
    uint k = line / N;
    uint i = line - k * N;

    threadgroup float2 buf[1024];
    uint logN = log2_uint(N);

    if (j < N) {
        uint src = (k * N + j) * N + i;
        buf[bit_reverse(j, logN)] = in_data[src];
    }
    line_fft_butterflies(buf, j, N, logN);
    if (j < N) {
        uint dst = (k * N + j) * N + i;
        out_data[dst] = buf[j];
    }
}

// FFT along z (stride N*N). Line index gid.y decodes (i, j); element gid.x = k.
kernel void fft3d_z(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint k = gid.x;
    uint line = gid.y;
    uint j = line / N;
    uint i = line - j * N;

    threadgroup float2 buf[1024];
    uint logN = log2_uint(N);

    if (k < N) {
        uint src = (k * N + j) * N + i;
        buf[bit_reverse(k, logN)] = in_data[src];
    }
    line_fft_butterflies(buf, k, N, logN);
    if (k < N) {
        uint dst = (k * N + j) * N + i;
        out_data[dst] = buf[k];
    }
}
