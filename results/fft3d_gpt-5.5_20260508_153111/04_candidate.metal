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

inline float2 mul_i(float2 a) {
    return float2(-a.y, a.x);
}

inline float2 mul_neg_i(float2 a) {
    return float2(a.y, -a.x);
}

inline float2 w128_full(uint idx) {
    if (idx < 64u) {
        return W128[idx];
    } else {
        float2 w = W128[idx - 64u];
        return float2(-w.x, -w.y);
    }
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

inline float2 reg_stage2_special(float2 x, uint tid) {
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

inline float2 fft32_from_natural(float2 natural, uint lane) {
    uint rev = bit_reverse_bits(lane, 5u);
    float2 x = shfl2(natural, ushort(rev));

    x = reg_stage1(x, lane);
    x = reg_stage2_special(x, lane);
    x = reg_stage_const(x, lane,  4u, 16u);
    x = reg_stage_const(x, lane,  8u,  8u);
    x = reg_stage_const(x, lane, 16u,  4u);
    return x;
}

inline float2 fft_first5_from_bitrev(float2 x, uint lane) {
    x = reg_stage1(x, lane);
    x = reg_stage2_special(x, lane);
    x = reg_stage_const(x, lane,  4u, 16u);
    x = reg_stage_const(x, lane,  8u,  8u);
    x = reg_stage_const(x, lane, 16u,  4u);
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
    uint lane = tid & 31u;
    uint rev = bit_reverse_bits(tid, 6u);
    float2 x = in_data[base + rev * stride];

    x = fft_first5_from_bitrev(x, lane);

    buf[tid] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint kk = lane;
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

    // After these 5 stages, slots 0,1,2,3 contain 32-point FFTs of
    // input residues 0,2,1,3 modulo 4 respectively.
    x = fft_first5_from_bitrev(x, lane);

    buf[tid] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float2 w1 = W128[lane];          // W_128^r
    float2 w2 = W128[lane << 1];     // W_128^(2r)
    float2 w3 = w128_full(3u * lane);// W_128^(3r)

    float2 out;

    if (s == 0u) {
        float2 f0 = x;
        float2 f1 = buf[32u + lane]; // residue 2
        float2 f2 = buf[64u + lane]; // residue 1
        float2 f3 = buf[96u + lane]; // residue 3

        float2 b1 = cmul(w2, f1);
        float2 b2 = cmul(w1, f2);
        float2 b3 = cmul(w3, f3);
        out = f0 + b1 + b2 + b3;
    } else if (s == 1u) {
        float2 f0 = buf[lane];
        float2 f1 = x;              // residue 2
        float2 f2 = buf[64u + lane];// residue 1
        float2 f3 = buf[96u + lane];// residue 3

        float2 b1 = cmul(w2, f1);
        float2 b2 = cmul(w1, f2);
        float2 b3 = cmul(w3, f3);
        out = f0 - b1 + mul_neg_i(b2) + mul_i(b3);
    } else if (s == 2u) {
        float2 f0 = buf[lane];
        float2 f1 = buf[32u + lane];// residue 2
        float2 f2 = x;              // residue 1
        float2 f3 = buf[96u + lane];// residue 3

        float2 b1 = cmul(w2, f1);
        float2 b2 = cmul(w1, f2);
        float2 b3 = cmul(w3, f3);
        out = f0 + b1 - b2 - b3;
    } else {
        float2 f0 = buf[lane];
        float2 f1 = buf[32u + lane];// residue 2
        float2 f2 = buf[64u + lane];// residue 1
        float2 f3 = x;              // residue 3

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