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

inline float2 mul_neg_i(float2 v) {
    return float2(v.y, -v.x);
}

inline float2 shfl2(float2 v, ushort lane) {
    return float2(simd_shuffle(v.x, lane), simd_shuffle(v.y, lane));
}

inline float2 shflxor2(float2 v, ushort mask) {
    return float2(simd_shuffle_xor(v.x, mask), simd_shuffle_xor(v.y, mask));
}

inline uint brev5(uint x) {
    return ((x &  1u) << 4) |
           ((x &  2u) << 2) |
            (x &  4u)       |
           ((x &  8u) >> 2) |
           ((x & 16u) >> 4);
}

inline uint brev6(uint x) {
    return (brev5(x & 31u) << 1) | (x >> 5);
}

inline uint brev7(uint x) {
    uint g = x >> 5;
    return (brev5(x & 31u) << 2) | ((g & 1u) << 1) | (g >> 1);
}

// DIT stages: bit-reversed input -> natural output.
inline float2 reg_stage1(float2 x, uint tid) {
    float2 y = shflxor2(x, ushort(1));
    return ((tid & 1u) == 0u) ? (x + y) : (y - x);
}

inline float2 reg_stage2(float2 x, uint tid) {
    float2 y = shflxor2(x, ushort(2));
    bool lo = ((tid & 2u) == 0u);
    bool k1 = ((tid & 1u) != 0u);
    if (lo) {
        float2 t = k1 ? mul_neg_i(y) : y;
        return x + t;
    } else {
        float2 t = k1 ? mul_neg_i(x) : x;
        return y - t;
    }
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
    float2 x = shfl2(natural, ushort(brev5(tid)));
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

inline void fft_line_32_io(device const float2 *in_data,
                           device       float2 *out_data,
                           uint in_base,
                           uint in_stride,
                           uint out_base,
                           uint out_stride,
                           uint tid) {
    float2 natural = in_data[in_base + tid * in_stride];
    float2 x = fft32_from_natural(natural, tid);
    out_data[out_base + tid * out_stride] = x;
}

inline void fft_line_64_io(device const float2 *in_data,
                           device       float2 *out_data,
                           uint in_base,
                           uint in_stride,
                           uint out_base,
                           uint out_stride,
                           uint tid,
                           threadgroup float2 *buf) {
    float2 x = in_data[in_base + brev6(tid) * in_stride];

    x = fft_first5_from_bitrev(x, tid);

    buf[tid] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint kk = tid & 31u;
    bool lo = ((tid & 32u) == 0u);
    float2 other = buf[tid ^ 32u];
    float2 w = W128[kk << 1];
    float2 t = cmul(w, lo ? other : x);
    x = lo ? (x + t) : (other - t);

    out_data[out_base + tid * out_stride] = x;
}

inline void fft_line_128_io(device const float2 *in_data,
                            device       float2 *out_data,
                            uint in_base,
                            uint in_stride,
                            uint out_base,
                            uint out_stride,
                            uint tid,
                            threadgroup float2 *buf0,
                            threadgroup float2 *buf1) {
    float2 x = in_data[in_base + brev7(tid) * in_stride];

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

    out_data[out_base + tid * out_stride] = x;
}

// DIF stages: natural input -> bit-reversed output coordinate.
inline float2 dif_stage1(float2 x, uint tid) {
    float2 y = shflxor2(x, ushort(1));
    return ((tid & 1u) == 0u) ? (x + y) : (y - x);
}

inline float2 dif_stage2(float2 x, uint tid) {
    float2 y = shflxor2(x, ushort(2));
    if ((tid & 2u) == 0u) {
        return x + y;
    } else {
        float2 d = y - x;
        return ((tid & 1u) != 0u) ? mul_neg_i(d) : d;
    }
}

inline float2 dif_stage_const(float2 x, uint tid, uint hspan, uint step128) {
    float2 y = shflxor2(x, ushort(hspan));
    if ((tid & hspan) == 0u) {
        return x + y;
    } else {
        uint kk = tid & (hspan - 1u);
        return cmul(y - x, W128[kk * step128]);
    }
}

inline float2 dif_tg_stage(float2 x,
                           uint tid,
                           uint hspan,
                           uint step128,
                           threadgroup float2 *buf) {
    buf[tid] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float2 y = buf[tid ^ hspan];
    if ((tid & hspan) == 0u) {
        return x + y;
    } else {
        uint kk = tid & (hspan - 1u);
        return cmul(y - x, W128[kk * step128]);
    }
}

inline void fft_line_32_dif_bitrev_io(device const float2 *in_data,
                                      device       float2 *out_data,
                                      uint in_base,
                                      uint in_stride,
                                      uint out_base,
                                      uint out_stride,
                                      uint tid) {
    float2 x = in_data[in_base + tid * in_stride];

    x = dif_stage_const(x, tid, 16u,  4u);
    x = dif_stage_const(x, tid,  8u,  8u);
    x = dif_stage_const(x, tid,  4u, 16u);
    x = dif_stage2(x, tid);
    x = dif_stage1(x, tid);

    out_data[out_base + tid * out_stride] = x;
}

inline void fft_line_64_dif_bitrev_io(device const float2 *in_data,
                                      device       float2 *out_data,
                                      uint in_base,
                                      uint in_stride,
                                      uint out_base,
                                      uint out_stride,
                                      uint tid,
                                      threadgroup float2 *buf) {
    float2 x = in_data[in_base + tid * in_stride];

    x = dif_tg_stage(x, tid, 32u, 2u, buf);
    x = dif_stage_const(x, tid, 16u,  4u);
    x = dif_stage_const(x, tid,  8u,  8u);
    x = dif_stage_const(x, tid,  4u, 16u);
    x = dif_stage2(x, tid);
    x = dif_stage1(x, tid);

    out_data[out_base + tid * out_stride] = x;
}

inline void fft_line_128_dif_bitrev_io(device const float2 *in_data,
                                       device       float2 *out_data,
                                       uint in_base,
                                       uint in_stride,
                                       uint out_base,
                                       uint out_stride,
                                       uint tid,
                                       threadgroup float2 *buf0,
                                       threadgroup float2 *buf1) {
    float2 x = in_data[in_base + tid * in_stride];

    x = dif_tg_stage(x, tid, 64u, 1u, buf0);
    x = dif_tg_stage(x, tid, 32u, 2u, buf1);
    x = dif_stage_const(x, tid, 16u,  4u);
    x = dif_stage_const(x, tid,  8u,  8u);
    x = dif_stage_const(x, tid,  4u, 16u);
    x = dif_stage2(x, tid);
    x = dif_stage1(x, tid);

    out_data[out_base + tid * out_stride] = x;
}

inline void fft_line_direct_fallback_io(device const float2 *in_data,
                                        device       float2 *out_data,
                                        uint in_base,
                                        uint in_stride,
                                        uint out_base,
                                        uint out_stride,
                                        uint tid,
                                        uint N) {
    float2 acc = float2(0.0f, 0.0f);
    float theta = -TWO_PI * float(tid) / float(N);
    float2 wstep = float2(cos(theta), sin(theta));
    float2 w = float2(1.0f, 0.0f);

    for (uint n = 0u; n < N; ++n) {
        float2 v = in_data[in_base + n * in_stride];
        acc += cmul(v, w);
        w = cmul(w, wstep);
    }

    out_data[out_base + tid * out_stride] = acc;
}

// Pass 1: logical X FFT.
// Fast path writes X-frequency in bit-reversed coordinate ixp:
// L1 index = k + N*ixp + N^2*j, value uses i = bitrev(ixp).
kernel void fft3d_x(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint tid = gid.x;
    uint line = gid.y;

    threadgroup float2 buf0[128];
    threadgroup float2 buf1[128];

    if (N == 32u) {
        uint j = line >> 5;
        uint k = line & 31u;
        uint in_base  = (k << 10) + (j << 5);
        uint out_base = k + (j << 10);
        fft_line_32_dif_bitrev_io(in_data, out_data, in_base, 1u, out_base, 32u, tid);
    } else if (N == 64u) {
        uint j = line >> 6;
        uint k = line & 63u;
        uint in_base  = (k << 12) + (j << 6);
        uint out_base = k + (j << 12);
        fft_line_64_dif_bitrev_io(in_data, out_data, in_base, 1u, out_base, 64u, tid, buf0);
    } else if (N == 128u) {
        uint j = line >> 7;
        uint k = line & 127u;
        uint in_base  = (k << 14) + (j << 7);
        uint out_base = k + (j << 14);
        fft_line_128_dif_bitrev_io(in_data, out_data, in_base, 1u, out_base, 128u, tid, buf0, buf1);
    } else {
        uint j = line / N;
        uint k = line - j * N;
        uint N2 = N * N;
        uint in_base  = (k * N + j) * N;
        uint out_base = k + N2 * j;
        fft_line_direct_fallback_io(in_data, out_data, in_base, 1u, out_base, N, tid, N);
    }
}

// Pass 2: logical Z FFT.
// Fast path preserves bit-reversed X coordinate ixp and writes
// bit-reversed Z coordinate kp:
// L2 index = j + N*kp + N^2*ixp.
kernel void fft3d_y(device const float2 *in_data  [[buffer(0)]],
                    device       float2 *out_data [[buffer(1)]],
                    constant uint        &N        [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint tid = gid.x;
    uint line = gid.y;

    threadgroup float2 buf0[128];
    threadgroup float2 buf1[128];

    if (N == 32u) {
        uint ixp = line >> 5;
        uint j   = line & 31u;
        uint in_base  = (ixp << 5) + (j << 10);
        uint out_base = j + (ixp << 10);
        fft_line_32_dif_bitrev_io(in_data, out_data, in_base, 1u, out_base, 32u, tid);
    } else if (N == 64u) {
        uint ixp = line >> 6;
        uint j   = line & 63u;
        uint in_base  = (ixp << 6) + (j << 12);
        uint out_base = j + (ixp << 12);
        fft_line_64_dif_bitrev_io(in_data, out_data, in_base, 1u, out_base, 64u, tid, buf0);
    } else if (N == 128u) {
        uint ixp = line >> 7;
        uint j   = line & 127u;
        uint in_base  = (ixp << 7) + (j << 14);
        uint out_base = j + (ixp << 14);
        fft_line_128_dif_bitrev_io(in_data, out_data, in_base, 1u, out_base, 128u, tid, buf0, buf1);
    } else {
        uint i = line / N;
        uint j = line - i * N;
        uint N2 = N * N;
        uint in_base  = N * i + N2 * j;
        uint out_base = j + N2 * i;
        fft_line_direct_fallback_io(in_data, out_data, in_base, 1u, out_base, N, tid, N);
    }
}

// Pass 3: logical Y FFT, mapping bit-reversed X/Z line coordinates
// back to final natural row-major [i fastest, j, k].
kernel void fft3d_z(device const float2 *in_data  [[buffer(0)]],
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
        uint kp  = brev5(k);
        uint ixp = brev5(i);
        uint in_base  = (kp << 5) + (ixp << 10);
        uint out_base = i + (k << 10);
        fft_line_32_io(in_data, out_data, in_base, 1u, out_base, 32u, tid);
    } else if (N == 64u) {
        uint k = line >> 6;
        uint i = line & 63u;
        uint kp  = brev6(k);
        uint ixp = brev6(i);
        uint in_base  = (kp << 6) + (ixp << 12);
        uint out_base = i + (k << 12);
        fft_line_64_io(in_data, out_data, in_base, 1u, out_base, 64u, tid, buf0);
    } else if (N == 128u) {
        uint k = line >> 7;
        uint i = line & 127u;
        uint kp  = brev7(k);
        uint ixp = brev7(i);
        uint in_base  = (kp << 7) + (ixp << 14);
        uint out_base = i + (k << 14);
        fft_line_128_io(in_data, out_data, in_base, 1u, out_base, 128u, tid, buf0, buf1);
    } else {
        uint k = line / N;
        uint i = line - k * N;
        uint N2 = N * N;
        uint in_base  = N * k + N2 * i;
        uint out_base = i + N2 * k;
        fft_line_direct_fallback_io(in_data, out_data, in_base, 1u, out_base, N, tid, N);
    }
}