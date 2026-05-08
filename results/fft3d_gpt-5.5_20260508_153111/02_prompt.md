## Task: fft3d

3D complex-to-complex forward FFT, fp32, on a power-of-two cube of side N. Convention: forward, unnormalized — 
  Y[k,j,i] = sum_{kk,jj,ii} X[kk,jj,ii] * exp(-2πi (k·kk + j·jj + i·ii) / N)
(matches numpy.fft.fftn with norm='backward').

Storage is row-major float2[NZ][NY][NX] with NX=NY=NZ=N. Linear index of element (i,j,k) is ((k·N + j)·N + i); float2 is (real, imag) and is the buffer element type. The host calls three separate kernels — fft3d_x, fft3d_y, fft3d_z — in that order, ping-ponging between two device buffers (so the final 3D FFT result lands in the second buffer). Each kernel does one 1D length-N FFT per threadgroup; the FFT axis is fixed by the kernel name and its index decoding.

Because the three axes are orthogonal, the FFTs commute — the result is invariant to the order x→y→z vs any other order, but the host fixes the order x→y→z and the kernel names must match. The optimization surface is dominated by data movement: bit-reversal vs Stockham auto-sort, twiddle caching, simdgroup-shuffle butterflies, and threadgroup-memory bank-conflict avoidance are all on the table.

## Required kernel signature(s)

```
kernel void fft3d_x(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]);
kernel void fft3d_y(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]);
kernel void fft3d_z(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]);

Dispatch geometry (identical for all three kernels, host-fixed):
  threadsPerGrid        = (N, N*N, 1)
  threadsPerThreadgroup = (N, 1,   1)
So each TG of N threads owns exactly one length-N line: gid.x is the position along the FFT axis (= thread_position_in_threadgroup.x) and gid.y indexes the (N×N) plane of lines orthogonal to that axis. Index decoding per kernel:
  fft3d_x: i = gid.x; k = gid.y / N; j = gid.y - k*N
  fft3d_y: j = gid.x; k = gid.y / N; i = gid.y - k*N
  fft3d_z: k = gid.x; j = gid.y / N; i = gid.y - j*N
Each TG must produce the full FFT of its line in out_data; the host runs the three kernels back-to-back in one command buffer and ping-pongs the buffers, so out_data of one pass is the in_data of the next.

If you cap the threadgroup with [[max_total_threads_per_threadgroup(N)]], place the attribute on the kernel declaration line itself (not as a free-standing statement), and remember the host's TG width along x is N (≤ 1024 on M-series). Your tile / shared-memory layout MUST match the dispatched (N, 1, 1) TG geometry: the host will not split or reshape the dispatch to fit a different tile.
```

## Your previous attempt

```metal
#include <metal_stdlib>
using namespace metal;

constant float TWO_PI = 6.28318530717958647692f;

constant float2 W128[64] = {
    float2( 1.000000000f, -0.000000000f),
    float2( 0.998795456f, -0.049067674f),
    float2( 0.995184727f, -0.098017140f),
    float2( 0.989176510f, -0.146730474f),
    float2( 0.980785280f, -0.195090322f),
    float2( 0.970031253f, -0.242980180f),
    float2( 0.956940336f, -0.290284677f),
    float2( 0.941544065f, -0.336889853f),
    float2( 0.923879533f, -0.382683432f),
    float2( 0.903989293f, -0.427555093f),
    float2( 0.881921264f, -0.471396737f),
    float2( 0.857728610f, -0.514102744f),
    float2( 0.831469612f, -0.555570233f),
    float2( 0.803207531f, -0.595699304f),
    float2( 0.773010453f, -0.634393284f),
    float2( 0.740951125f, -0.671558955f),
    float2( 0.707106781f, -0.707106781f),
    float2( 0.671558955f, -0.740951125f),
    float2( 0.634393284f, -0.773010453f),
    float2( 0.595699304f, -0.803207531f),
    float2( 0.555570233f, -0.831469612f),
    float2( 0.514102744f, -0.857728610f),
    float2( 0.471396737f, -0.881921264f),
    float2( 0.427555093f, -0.903989293f),
    float2( 0.382683432f, -0.923879533f),
    float2( 0.336889853f, -0.941544065f),
    float2( 0.290284677f, -0.956940336f),
    float2( 0.242980180f, -0.970031253f),
    float2( 0.195090322f, -0.980785280f),
    float2( 0.146730474f, -0.989176510f),
    float2( 0.098017140f, -0.995184727f),
    float2( 0.049067674f, -0.998795456f),
    float2( 0.000000000f, -1.000000000f),
    float2(-0.049067674f, -0.998795456f),
    float2(-0.098017140f, -0.995184727f),
    float2(-0.146730474f, -0.989176510f),
    float2(-0.195090322f, -0.980785280f),
    float2(-0.242980180f, -0.970031253f),
    float2(-0.290284677f, -0.956940336f),
    float2(-0.336889853f, -0.941544065f),
    float2(-0.382683432f, -0.923879533f),
    float2(-0.427555093f, -0.903989293f),
    float2(-0.471396737f, -0.881921264f),
    float2(-0.514102744f, -0.857728610f),
    float2(-0.555570233f, -0.831469612f),
    float2(-0.595699304f, -0.803207531f),
    float2(-0.634393284f, -0.773010453f),
    float2(-0.671558955f, -0.740951125f),
    float2(-0.707106781f, -0.707106781f),
    float2(-0.740951125f, -0.671558955f),
    float2(-0.773010453f, -0.634393284f),
    float2(-0.803207531f, -0.595699304f),
    float2(-0.831469612f, -0.555570233f),
    float2(-0.857728610f, -0.514102744f),
    float2(-0.881921264f, -0.471396737f),
    float2(-0.903989293f, -0.427555093f),
    float2(-0.923879533f, -0.382683432f),
    float2(-0.941544065f, -0.336889853f),
    float2(-0.956940336f, -0.290284677f),
    float2(-0.970031253f, -0.242980180f),
    float2(-0.980785280f, -0.195090322f),
    float2(-0.989176510f, -0.146730474f),
    float2(-0.995184727f, -0.098017140f),
    float2(-0.998795456f, -0.049067674f)
};

inline uint log2_pow2(uint n) {
    if (n == 32u)  return 5u;
    if (n == 64u)  return 6u;
    if (n == 128u) return 7u;
    uint r = 0u;
    while (n > 1u) {
        n >>= 1u;
        ++r;
    }
    return r;
}

inline uint bit_reverse_fast(uint x, uint logN) {
    x = ((x & 0x55555555u) << 1) | ((x >> 1) & 0x55555555u);
    x = ((x & 0x33333333u) << 2) | ((x >> 2) & 0x33333333u);
    x = ((x & 0x0f0f0f0fu) << 4) | ((x >> 4) & 0x0f0f0f0fu);
    x = ((x & 0x00ff00ffu) << 8) | ((x >> 8) & 0x00ff00ffu);
    x = (x << 16) | (x >> 16);
    return (logN == 0u) ? 0u : (x >> (32u - logN));
}

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y,
                  a.x * b.y + a.y * b.x);
}

inline float2 shuffle_xor_float2(float2 v, ushort mask) {
    return float2(simd_shuffle_xor(v.x, mask),
                  simd_shuffle_xor(v.y, mask));
}

inline float2 reg_stage_1(float2 x, uint tid) {
    float2 y = shuffle_xor_float2(x, ushort(1));
    return ((tid & 1u) == 0u) ? (x + y) : (y - x);
}

inline float2 reg_stage(float2 x, uint tid, uint half, uint step128) {
    float2 y = shuffle_xor_float2(x, ushort(half));
    uint kk = tid & (half - 1u);
    bool lo = ((tid & half) == 0u);
    float2 w = W128[kk * step128];
    float2 t = cmul(w, lo ? y : x);
    return lo ? (x + t) : (y - t);
}

inline float2 fft_first_five_stages(float2 x, uint tid, uint logN) {
    if (logN >= 1u) x = reg_stage_1(x, tid);
    if (logN >= 2u) x = reg_stage(x, tid,  2u, 32u);
    if (logN >= 3u) x = reg_stage(x, tid,  4u, 16u);
    if (logN >= 4u) x = reg_stage(x, tid,  8u,  8u);
    if (logN >= 5u) x = reg_stage(x, tid, 16u,  4u);
    return x;
}

inline void fft_remaining_threadgroup(threadgroup float2 *buf,
                                      uint tid,
                                      uint N,
                                      uint logN) {
    for (uint s = 6u; s <= logN; ++s) {
        uint m = 1u << s;
        uint half = m >> 1u;

        if (tid < (N >> 1u)) {
            uint kk = tid & (half - 1u);
            uint base = (tid - kk) << 1;

            float2 w;
            if (s <= 7u) {
                w = W128[kk * (128u >> s)];
            } else {
                float angle = -TWO_PI * float(kk) / float(m);
                w = float2(cos(angle), sin(angle));
            }

            float2 u = buf[base + kk];
            float2 v = buf[base + half + kk];
            float2 t = cmul(w, v);

            buf[base + kk]        = u + t;
            buf[base + half + kk] = u - t;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

inline void fft_line_direct(device const float2 *in_data,
                            device       float2 *out_data,
                            uint base,
                            uint stride,
                            uint tid,
                            uint N,
                            threadgroup float2 *buf) {
    uint logN = log2_pow2(N);
    uint rev = bit_reverse_fast(tid, logN);

    float2 x = in_data[base + rev * stride];

    x = fft_first_five_stages(x, tid, logN);

    if (logN > 5u) {
        buf[tid] = x;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        fft_remaining_threadgroup(buf, tid, N, logN);

        x = buf[tid];
    }

    out_data[base + tid * stride] = x;
}

kernel void fft3d_x(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint tid = gid.x;
    uint line = gid.y;

    threadgroup float2 buf[1024];

    uint base = line * N;
    fft_line_direct(in_data, out_data, base, 1u, tid, N, buf);
}

kernel void fft3d_y(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint tid = gid.x;
    uint line = gid.y;

    uint k = line / N;
    uint i = line - k * N;

    threadgroup float2 buf[1024];

    uint base = k * N * N + i;
    fft_line_direct(in_data, out_data, base, N, tid, N, buf);
}

kernel void fft3d_z(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint tid = gid.x;
    uint line = gid.y;

    threadgroup float2 buf[1024];

    uint base = line;
    uint stride = N * N;
    fft_line_direct(in_data, out_data, base, stride, tid, N, buf);
}
```

Result of previous attempt:
  COMPILE FAILED: Error Domain=MTLLibraryErrorDomain Code=3 "program_source:109:50: error: cannot combine with previous 'type-name' declaration specifier
inline float2 reg_stage(float2 x, uint tid, uint half, uint step128) {
                                                 ^
program_source:110:49: error: expected '(' for function-style cast or type construction
    float2 y = shuffle_xor_float2(x, ushort(half));
                                            ~~~~^
program_source:111:27: error: expected ')'
    uint kk = tid & (half - 1u);
                          ^
program_source:111:21: note: to match this '('
    uint kk = tid & (half - 1u);
                    ^
program_source:111:32: error: expected expression
    uint kk = tid & (half - 1u);
                               ^
program_source:112:27: error: expected '(' for function-style cast or type construction
    bool lo = ((tid & half) == 0u);
                      ~~~~^
program_source:133:14: error: cannot combine with previous 'type-name' declaration specifier
        uint half = m >> 1u;
             ^
program_source:133:19: error: expected unqualified-id
        uint half = m >> 1u;
                  ^
program_source:136:35: error: expected ')'
            uint kk = tid & (half - 1u);
                                  ^
program_source:136:29: note: to match this '('
            uint kk = tid & (half - 1u);
                            ^
program_source:136:40: error: expected expression
            uint kk = tid & (half - 1u);
                                       ^
program_source:148:40: error: expected '(' for function-style cast or type construction
            float2 v = buf[base + half + kk];
                                  ~~~~ ^
program_source:152:29: error: expected '(' for function-style cast or type construction
            buf[base + half + kk] = u - t;
                       ~~~~ ^
" UserInfo={NSLocalizedDescription=program_source:109:50: error: cannot combine with previous 'type-name' declaration specifier
inline float2 reg_stage(float2 x, uint tid, uint half, uint step128) {
                                                 ^
program_source:110:49: error: expected '(' for function-style cast or type construction
    float2 y = shuffle_xor_float2(x, ushort(half));
                                            ~~~~^
program_source:111:27: error: expected ')'
    uint kk = tid & (half - 1u);
                          ^
program_source:111:21: note: to match this '('
    uint kk = tid & (half - 1u);
                    ^
program_source:111:32: error: expected expression
    uint kk = tid & (half - 1u);
                               ^
program_source:112:27: error: expected '(' for function-style cast or type construction
    bool lo = ((tid & half) == 0u);
                      ~~~~^
program_source:133:14: error: cannot combine with previous 'type-name' declaration specifier
        uint half = m >> 1u;
             ^
program_source:133:19: error: expected unqualified-id
        uint half = m >> 1u;
                  ^
program_source:136:35: error: expected ')'
            uint kk = tid & (half - 1u);
                                  ^
program_source:136:29: note: to match this '('
            uint kk = tid & (half - 1u);
                            ^
program_source:136:40: error: expected expression
            uint kk = tid & (half - 1u);
                                       ^
program_source:148:40: error: expected '(' for function-style cast or type construction
            float2 v = buf[base + half + kk];
                                  ~~~~ ^
program_source:152:29: error: expected '(' for function-style cast or type construction
            buf[base + half + kk] = u - t;
                       ~~~~ ^
}

## Current best (incumbent)

```metal
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
```

Incumbent result:
            32cube: correct, 0.22 ms, 14.0 GB/s (effective, 96 B/cell across 3 passes) (7.0% of 200 GB/s)
            64cube: correct, 0.93 ms, 27.1 GB/s (effective, 96 B/cell across 3 passes) (13.6% of 200 GB/s)
           128cube: correct, 2.80 ms, 71.9 GB/s (effective, 96 B/cell across 3 passes) (35.9% of 200 GB/s)
  score (gmean of fraction): 0.1507

## History

- iter  0: compile=OK | correct=True | score=0.15065668929082848
- iter  1: compile=FAIL | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
