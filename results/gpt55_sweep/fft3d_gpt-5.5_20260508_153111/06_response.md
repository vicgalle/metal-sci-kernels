I’m specializing the fixed-size bit reversal used by the 32-lane SIMD FFT: instead of reversing all 32 bits, the code reverses only the 5 lane bits and composes 64/128-point reversals from SIMDgroup/lane bits. I also special-case the stage-2 `-i` twiddle to remove a constant-table load and complex multiply. This should reduce integer/ALU overhead on all three benchmark sizes while preserving the incumbent’s faster two-barrier 128-point path.

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

inline float2 mul_neg_i(float2 a) {
    return float2(a.y, -a.x);
}

inline float2 shfl2(float2 v, ushort lane) {
    return float2(simd_shuffle(v.x, lane), simd_shuffle(v.y, lane));
}

inline float2 shflxor2(float2 v, ushort mask) {
    return float2(simd_shuffle_xor(v.x, mask), simd_shuffle_xor(v.y, mask));
}

inline uint bit_reverse_5(uint x) {
    return ((x &  1u) << 4) |
           ((x &  2u) << 2) |
           ( x &  4u)       |
           ((x &  8u) >> 2) |
           ((x & 16u) >> 4);
}

inline uint bit_reverse_6_from_tid(uint tid) {
    uint lane = tid & 31u;
    uint sg   = tid >> 5;
    return (bit_reverse_5(lane) << 1) | sg;
}

inline uint bit_reverse_7_from_tid(uint tid) {
    uint lane = tid & 31u;
    uint sg   = tid >> 5;
    uint rsg  = ((sg & 1u) << 1) | (sg >> 1);
    return (bit_reverse_5(lane) << 2) | rsg;
}

inline float2 reg_stage1(float2 x, uint tid) {
    float2 y = shflxor2(x, ushort(1));
    return ((tid & 1u) == 0u) ? (x + y) : (y - x);
}

inline float2 reg_stage2(float2 x, uint tid) {
    float2 y = shflxor2(x, ushort(2));
    bool lo = ((tid & 2u) == 0u);
    float2 v = lo ? y : x;
    float2 t = ((tid & 1u) == 0u) ? v : mul_neg_i(v);
    return lo ? (x + t) : (y - t);
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
    uint rev = bit_reverse_5(tid);
    float2 x = shfl2(natural, ushort(rev));

    x = reg_stage1(x, tid);
    x = reg_stage2(x, tid);
    x = reg_stage_const(x, tid,  4u, 16u);
    x = reg_stage_const(x, tid,  8u,  8u);
    x = reg_stage_const(x, tid, 16u,  4u);
    return x;
}

inline float2 fft_first5_from_bitrev(float2 x, uint tid) {
    x = reg_stage1(x, tid);
    x = reg_stage2(x, tid);
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
    uint rev = bit_reverse_6_from_tid(tid);
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
    uint rev = bit_reverse_7_from_tid(tid);
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