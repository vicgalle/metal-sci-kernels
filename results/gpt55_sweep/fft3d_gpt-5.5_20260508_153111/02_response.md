I fix the compile failure by avoiding `half` as an identifier. The kernel now performs the first five FFT stages in SIMD registers using shuffle-xor, uses table twiddles for common N≤128, and computes the final shared-memory stage directly into each thread’s output, removing one barrier and shared write pass versus the incumbent.

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

inline float2 shuffle_float2(float2 v, ushort lane) {
    return float2(simd_shuffle(v.x, lane),
                  simd_shuffle(v.y, lane));
}

inline float2 shuffle_xor_float2(float2 v, ushort mask) {
    return float2(simd_shuffle_xor(v.x, mask),
                  simd_shuffle_xor(v.y, mask));
}

inline float2 twiddle_stage(uint kk, uint stage) {
    if (stage <= 7u) {
        return W128[kk * (128u >> stage)];
    } else {
        uint span = 1u << stage;
        float angle = -TWO_PI * float(kk) / float(span);
        return float2(cos(angle), sin(angle));
    }
}

inline float2 reg_stage_1(float2 x, uint tid) {
    float2 y = shuffle_xor_float2(x, ushort(1));
    return ((tid & 1u) == 0u) ? (x + y) : (y - x);
}

inline float2 reg_stage(float2 x, uint tid, uint hspan, uint step128) {
    float2 y = shuffle_xor_float2(x, ushort(hspan));
    uint kk = tid & (hspan - 1u);
    bool lo = ((tid & hspan) == 0u);
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

inline void fft_middle_shared_stages(threadgroup float2 *buf,
                                     uint tid,
                                     uint N,
                                     uint logN) {
    for (uint stage = 6u; stage < logN; ++stage) {
        uint span  = 1u << stage;
        uint hspan = span >> 1u;

        if (tid < (N >> 1u)) {
            uint kk = tid & (hspan - 1u);
            uint base_idx = (tid - kk) << 1;

            float2 w = twiddle_stage(kk, stage);
            float2 u = buf[base_idx + kk];
            float2 v = buf[base_idx + hspan + kk];
            float2 t = cmul(w, v);

            buf[base_idx + kk]         = u + t;
            buf[base_idx + hspan + kk] = u - t;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

inline float2 fft_final_shared_stage(threadgroup float2 *buf,
                                     uint tid,
                                     uint N,
                                     uint logN) {
    uint hspan = N >> 1u;
    uint kk = tid & (hspan - 1u);
    bool lo = ((tid & hspan) == 0u);

    float2 self  = buf[tid];
    float2 other = buf[tid ^ hspan];
    float2 w = twiddle_stage(kk, logN);
    float2 t = cmul(w, lo ? other : self);

    return lo ? (self + t) : (other - t);
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

    float2 x;

    if (logN <= 5u) {
        float2 natural = in_data[base + tid * stride];
        x = shuffle_float2(natural, ushort(rev));
        x = fft_first_five_stages(x, tid, logN);
        out_data[base + tid * stride] = x;
        return;
    }

    x = in_data[base + rev * stride];
    x = fft_first_five_stages(x, tid, logN);

    buf[tid] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    fft_middle_shared_stages(buf, tid, N, logN);

    x = fft_final_shared_stage(buf, tid, N, logN);
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