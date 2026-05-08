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

constant float2 W3_128[32] = {
    float2( 1.000000000f, -0.000000000f),
    float2( 0.989176510f, -0.146730474f),
    float2( 0.956940336f, -0.290284677f),
    float2( 0.903989293f, -0.427555093f),
    float2( 0.831469612f, -0.555570233f),
    float2( 0.740951125f, -0.671558955f),
    float2( 0.634393284f, -0.773010453f),
    float2( 0.514102744f, -0.857728610f),
    float2( 0.382683432f, -0.923879533f),
    float2( 0.242980180f, -0.970031253f),
    float2( 0.098017140f, -0.995184727f),
    float2(-0.049067674f, -0.998795456f),
    float2(-0.195090322f, -0.980785280f),
    float2(-0.336889853f, -0.941544065f),
    float2(-0.471396737f, -0.881921264f),
    float2(-0.595699304f, -0.803207531f),
    float2(-0.707106781f, -0.707106781f),
    float2(-0.803207531f, -0.595699304f),
    float2(-0.881921264f, -0.471396737f),
    float2(-0.941544065f, -0.336889853f),
    float2(-0.980785280f, -0.195090322f),
    float2(-0.998795456f, -0.049067674f),
    float2(-0.995184727f,  0.098017140f),
    float2(-0.970031253f,  0.242980180f),
    float2(-0.923879533f,  0.382683432f),
    float2(-0.857728610f,  0.514102744f),
    float2(-0.773010453f,  0.634393284f),
    float2(-0.671558955f,  0.740951125f),
    float2(-0.555570233f,  0.831469612f),
    float2(-0.427555093f,  0.903989293f),
    float2(-0.290284677f,  0.956940336f),
    float2(-0.146730474f,  0.989176510f)
};

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y,
                  a.x * b.y + a.y * b.x);
}

inline float2 mul_i(float2 a) {
    return float2(-a.y, a.x);
}

inline float2 mul_neg_i(float2 a) {
    return float2(a.y, -a.x);
}

inline float2 shfl2(float2 v, ushort lane) {
    return float2(simd_shuffle(v.x, lane), simd_shuffle(v.y, lane));
}

inline float2 shflxor2(float2 v, ushort mask) {
    return float2(simd_shuffle_xor(v.x, mask), simd_shuffle_xor(v.y, mask));
}

inline uint bit_reverse_bits(uint x, uint bits) {
    return reverse_bits(x) >> (32u - bits);
}

inline float2 reg_stage1(float2 x, uint tid) {
    float2 y = shflxor2(x, ushort(1));
    return ((tid & 1u) == 0u) ? (x + y) : (y - x);
}

inline float2 reg_stage_const(float2 x, uint tid, uint hspan, uint step128) {
    float2 y = shflxor2(x, ushort(hspan));
    uint kk = tid & (hspan - 1u);
    bool lo = ((tid & hspan) == 0u);
    float2 w = W128[kk * step128];
    float2 t = cmul(w, lo ? y : x);
    return lo ? (x + t) : (y - t);
}

inline float2 fft32_from_natural(float2 natural, uint tid) {
    uint rev = bit_reverse_bits(tid, 5u);
    float2 x = shfl2(natural, ushort(rev));

    x = reg_stage1(x, tid);
    x = reg_stage_const(x, tid,  2u, 32u);
    x = reg_stage_const(x, tid,  4u, 16u);
    x = reg_stage_const(x, tid,  8u,  8u);
    x = reg_stage_const(x, tid, 16u,  4u);
    return x;
}

inline float2 fft_first5_from_bitrev(float2 x, uint tid) {
    x = reg_stage1(x, tid);
    x = reg_stage_const(x, tid,  2u, 32u);
    x = reg_stage_const(x, tid,  4u, 16u);
    x = reg_stage_const(x, tid,  8u,  8u);
    x = reg_stage_const(x, tid, 16u,  4u);
    return x;
}

inline void fft_line_32(device const float2 *in_data,
                        device       float2 *out_data,
                        uint base,
                        uint stride,
                        uint tid) {
    float2 natural = in_data[base + tid * stride];
    float2 x = fft32_from_natural(natural, tid);
    out_data[base + tid * stride] = x;
}

inline void fft_line_64(device const float2 *in_data,
                        device       float2 *out_data,
                        uint base,
                        uint stride,
                        uint tid,
                        threadgroup float2 *buf) {
    uint rev = bit_reverse_bits(tid, 6u);
    float2 x = in_data[base + rev * stride];

    x = fft_first5_from_bitrev(x, tid);

    buf[tid] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint kk = tid & 31u;
    bool lo = ((tid & 32u) == 0u);
    float2 other = buf[tid ^ 32u];
    float2 w = W128[kk << 1];
    float2 t = cmul(w, lo ? other : x);
    x = lo ? (x + t) : (other - t);

    out_data[base + tid * stride] = x;
}

inline void fft_line_128(device const float2 *in_data,
                         device       float2 *out_data,
                         uint base,
                         uint stride,
                         uint tid,
                         threadgroup float2 *buf) {
    uint lane = tid & 31u;
    uint s = tid >> 5;

    uint rev = bit_reverse_bits(tid, 7u);
    float2 x = in_data[base + rev * stride];

    x = fft_first5_from_bitrev(x, tid);

    buf[tid] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float2 w1 = W128[lane];
    float2 w2 = W128[lane << 1];
    float2 w3 = W3_128[lane];

    float2 out;

    if (s == 0u) {
        float2 f0 = x;
        float2 f1 = buf[32u + lane];
        float2 f2 = buf[64u + lane];
        float2 f3 = buf[96u + lane];

        float2 b1 = cmul(w2, f1);
        float2 b2 = cmul(w1, f2);
        float2 b3 = cmul(w3, f3);
        out = f0 + b1 + b2 + b3;
    } else if (s == 1u) {
        float2 f0 = buf[lane];
        float2 f1 = x;
        float2 f2 = buf[64u + lane];
        float2 f3 = buf[96u + lane];

        float2 b1 = cmul(w2, f1);
        float2 b2 = cmul(w1, f2);
        float2 b3 = cmul(w3, f3);
        out = f0 - b1 + mul_neg_i(b2) + mul_i(b3);
    } else if (s == 2u) {
        float2 f0 = buf[lane];
        float2 f1 = buf[32u + lane];
        float2 f2 = x;
        float2 f3 = buf[96u + lane];

        float2 b1 = cmul(w2, f1);
        float2 b2 = cmul(w1, f2);
        float2 b3 = cmul(w3, f3);
        out = f0 + b1 - b2 - b3;
    } else {
        float2 f0 = buf[lane];
        float2 f1 = buf[32u + lane];
        float2 f2 = buf[64u + lane];
        float2 f3 = x;

        float2 b1 = cmul(w2, f1);
        float2 b2 = cmul(w1, f2);
        float2 b3 = cmul(w3, f3);
        out = f0 - b1 + mul_i(b2) + mul_neg_i(b3);
    }

    out_data[base + tid * stride] = out;
}

inline void fft_line_direct_fallback(device const float2 *in_data,
                                     device       float2 *out_data,
                                     uint base,
                                     uint stride,
                                     uint tid,
                                     uint N) {
    float2 acc = float2(0.0f, 0.0f);
    float theta = -TWO_PI * float(tid) / float(N);
    float2 wstep = float2(cos(theta), sin(theta));
    float2 w = float2(1.0f, 0.0f);

    for (uint n = 0u; n < N; ++n) {
        float2 v = in_data[base + n * stride];
        acc += cmul(v, w);
        w = cmul(w, wstep);
    }

    out_data[base + tid * stride] = acc;
}

kernel void fft3d_x(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint tid = gid.x;
    uint line = gid.y;

    threadgroup float2 buf[128];

    if (N == 32u) {
        fft_line_32(in_data, out_data, line << 5, 1u, tid);
    } else if (N == 64u) {
        fft_line_64(in_data, out_data, line << 6, 1u, tid, buf);
    } else if (N == 128u) {
        fft_line_128(in_data, out_data, line << 7, 1u, tid, buf);
    } else {
        fft_line_direct_fallback(in_data, out_data, line * N, 1u, tid, N);
    }
}

kernel void fft3d_y(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint tid = gid.x;
    uint line = gid.y;

    threadgroup float2 buf[128];

    if (N == 32u) {
        uint k = line >> 5;
        uint i = line & 31u;
        uint base = (k << 10) + i;
        fft_line_32(in_data, out_data, base, 32u, tid);
    } else if (N == 64u) {
        uint k = line >> 6;
        uint i = line & 63u;
        uint base = (k << 12) + i;
        fft_line_64(in_data, out_data, base, 64u, tid, buf);
    } else if (N == 128u) {
        uint k = line >> 7;
        uint i = line & 127u;
        uint base = (k << 14) + i;
        fft_line_128(in_data, out_data, base, 128u, tid, buf);
    } else {
        uint k = line / N;
        uint i = line - k * N;
        uint base = k * N * N + i;
        fft_line_direct_fallback(in_data, out_data, base, N, tid, N);
    }
}

kernel void fft3d_z(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint tid = gid.x;
    uint line = gid.y;

    threadgroup float2 buf[128];

    if (N == 32u) {
        fft_line_32(in_data, out_data, line, 1024u, tid);
    } else if (N == 64u) {
        fft_line_64(in_data, out_data, line, 4096u, tid, buf);
    } else if (N == 128u) {
        fft_line_128(in_data, out_data, line, 16384u, tid, buf);
    } else {
        fft_line_direct_fallback(in_data, out_data, line, N * N, tid, N);
    }
}
```

Result of previous attempt:
            32cube: correct, 0.08 ms, 40.5 GB/s (effective, 96 B/cell across 3 passes) (20.3% of 200 GB/s)
            64cube: correct, 0.37 ms, 68.6 GB/s (effective, 96 B/cell across 3 passes) (34.3% of 200 GB/s)
           128cube: correct, 1.61 ms, 125.1 GB/s (effective, 96 B/cell across 3 passes) (62.6% of 200 GB/s)
  score (gmean of fraction): 0.3517

## Current best (incumbent)

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

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y,
                  a.x * b.y + a.y * b.x);
}

inline float2 shfl2(float2 v, ushort lane) {
    return float2(simd_shuffle(v.x, lane), simd_shuffle(v.y, lane));
}

inline float2 shflxor2(float2 v, ushort mask) {
    return float2(simd_shuffle_xor(v.x, mask), simd_shuffle_xor(v.y, mask));
}

inline uint bit_reverse_bits(uint x, uint bits) {
    x = ((x & 0x55555555u) << 1) | ((x >> 1) & 0x55555555u);
    x = ((x & 0x33333333u) << 2) | ((x >> 2) & 0x33333333u);
    x = ((x & 0x0f0f0f0fu) << 4) | ((x >> 4) & 0x0f0f0f0fu);
    x = ((x & 0x00ff00ffu) << 8) | ((x >> 8) & 0x00ff00ffu);
    x = (x << 16) | (x >> 16);
    return x >> (32u - bits);
}

inline float2 reg_stage1(float2 x, uint tid) {
    float2 y = shflxor2(x, ushort(1));
    return ((tid & 1u) == 0u) ? (x + y) : (y - x);
}

inline float2 reg_stage_const(float2 x, uint tid, uint hspan, uint step128) {
    float2 y = shflxor2(x, ushort(hspan));
    uint kk = tid & (hspan - 1u);
    bool lo = ((tid & hspan) == 0u);
    float2 w = W128[kk * step128];
    float2 t = cmul(w, lo ? y : x);
    return lo ? (x + t) : (y - t);
}

inline float2 fft32_from_natural(float2 natural, uint tid) {
    uint rev = bit_reverse_bits(tid, 5u);
    float2 x = shfl2(natural, ushort(rev));

    x = reg_stage1(x, tid);
    x = reg_stage_const(x, tid,  2u, 32u);
    x = reg_stage_const(x, tid,  4u, 16u);
    x = reg_stage_const(x, tid,  8u,  8u);
    x = reg_stage_const(x, tid, 16u,  4u);
    return x;
}

inline float2 fft_first5_from_bitrev(float2 x, uint tid) {
    x = reg_stage1(x, tid);
    x = reg_stage_const(x, tid,  2u, 32u);
    x = reg_stage_const(x, tid,  4u, 16u);
    x = reg_stage_const(x, tid,  8u,  8u);
    x = reg_stage_const(x, tid, 16u,  4u);
    return x;
}

inline void fft_line_32(device const float2 *in_data,
                        device       float2 *out_data,
                        uint base,
                        uint stride,
                        uint tid) {
    float2 natural = in_data[base + tid * stride];
    float2 x = fft32_from_natural(natural, tid);
    out_data[base + tid * stride] = x;
}

inline void fft_line_64(device const float2 *in_data,
                        device       float2 *out_data,
                        uint base,
                        uint stride,
                        uint tid,
                        threadgroup float2 *buf) {
    uint rev = bit_reverse_bits(tid, 6u);
    float2 x = in_data[base + rev * stride];

    x = fft_first5_from_bitrev(x, tid);

    buf[tid] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint kk = tid & 31u;
    bool lo = ((tid & 32u) == 0u);
    float2 other = buf[tid ^ 32u];
    float2 w = W128[kk << 1];
    float2 t = cmul(w, lo ? other : x);
    x = lo ? (x + t) : (other - t);

    out_data[base + tid * stride] = x;
}

inline void fft_line_128(device const float2 *in_data,
                         device       float2 *out_data,
                         uint base,
                         uint stride,
                         uint tid,
                         threadgroup float2 *buf0,
                         threadgroup float2 *buf1) {
    uint rev = bit_reverse_bits(tid, 7u);
    float2 x = in_data[base + rev * stride];

    x = fft_first5_from_bitrev(x, tid);

    buf0[tid] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        uint kk = tid & 31u;
        bool lo = ((tid & 32u) == 0u);
        float2 other = buf0[tid ^ 32u];
        float2 w = W128[kk << 1];
        float2 t = cmul(w, lo ? other : x);
        x = lo ? (x + t) : (other - t);
    }

    buf1[tid] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        uint kk = tid & 63u;
        bool lo = ((tid & 64u) == 0u);
        float2 other = buf1[tid ^ 64u];
        float2 w = W128[kk];
        float2 t = cmul(w, lo ? other : x);
        x = lo ? (x + t) : (other - t);
    }

    out_data[base + tid * stride] = x;
}

inline void fft_line_direct_fallback(device const float2 *in_data,
                                     device       float2 *out_data,
                                     uint base,
                                     uint stride,
                                     uint tid,
                                     uint N) {
    float2 acc = float2(0.0f, 0.0f);
    float theta = -TWO_PI * float(tid) / float(N);
    float2 wstep = float2(cos(theta), sin(theta));
    float2 w = float2(1.0f, 0.0f);

    for (uint n = 0u; n < N; ++n) {
        float2 v = in_data[base + n * stride];
        acc += cmul(v, w);
        w = cmul(w, wstep);
    }

    out_data[base + tid * stride] = acc;
}

kernel void fft3d_x(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint tid = gid.x;
    uint line = gid.y;

    threadgroup float2 buf0[128];
    threadgroup float2 buf1[128];

    if (N == 32u) {
        fft_line_32(in_data, out_data, line << 5, 1u, tid);
    } else if (N == 64u) {
        fft_line_64(in_data, out_data, line << 6, 1u, tid, buf0);
    } else if (N == 128u) {
        fft_line_128(in_data, out_data, line << 7, 1u, tid, buf0, buf1);
    } else {
        fft_line_direct_fallback(in_data, out_data, line * N, 1u, tid, N);
    }
}

kernel void fft3d_y(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint tid = gid.x;
    uint line = gid.y;

    threadgroup float2 buf0[128];
    threadgroup float2 buf1[128];

    if (N == 32u) {
        uint k = line >> 5;
        uint i = line & 31u;
        uint base = (k << 10) + i;
        fft_line_32(in_data, out_data, base, 32u, tid);
    } else if (N == 64u) {
        uint k = line >> 6;
        uint i = line & 63u;
        uint base = (k << 12) + i;
        fft_line_64(in_data, out_data, base, 64u, tid, buf0);
    } else if (N == 128u) {
        uint k = line >> 7;
        uint i = line & 127u;
        uint base = (k << 14) + i;
        fft_line_128(in_data, out_data, base, 128u, tid, buf0, buf1);
    } else {
        uint k = line / N;
        uint i = line - k * N;
        uint base = k * N * N + i;
        fft_line_direct_fallback(in_data, out_data, base, N, tid, N);
    }
}

kernel void fft3d_z(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint tid = gid.x;
    uint line = gid.y;

    threadgroup float2 buf0[128];
    threadgroup float2 buf1[128];

    if (N == 32u) {
        fft_line_32(in_data, out_data, line, 1024u, tid);
    } else if (N == 64u) {
        fft_line_64(in_data, out_data, line, 4096u, tid, buf0);
    } else if (N == 128u) {
        fft_line_128(in_data, out_data, line, 16384u, tid, buf0, buf1);
    } else {
        fft_line_direct_fallback(in_data, out_data, line, N * N, tid, N);
    }
}
```

Incumbent result:
            32cube: correct, 0.08 ms, 41.7 GB/s (effective, 96 B/cell across 3 passes) (20.8% of 200 GB/s)
            64cube: correct, 0.35 ms, 72.6 GB/s (effective, 96 B/cell across 3 passes) (36.3% of 200 GB/s)
           128cube: correct, 1.57 ms, 128.0 GB/s (effective, 96 B/cell across 3 passes) (64.0% of 200 GB/s)
  score (gmean of fraction): 0.3644

## History

- iter  0: compile=OK | correct=True | score=0.15065668929082848
- iter  1: compile=FAIL | correct=False | score=N/A
- iter  2: compile=OK | correct=True | score=0.2294201441180084
- iter  3: compile=OK | correct=True | score=0.3644396895993623
- iter  4: compile=OK | correct=True | score=0.3605152630611077
- iter  5: compile=OK | correct=True | score=0.35172079898512837

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
