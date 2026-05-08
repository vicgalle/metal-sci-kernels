This version uses SIMDgroup shuffles to reuse `u_curr` values already loaded by neighboring lanes for X/Y/Z stencil neighbors when the threadgroup layout places those neighbors in the same SIMDgroup. Coordinate tags validate every shuffle, with safe fallback to global memory, preserving correctness for boundaries and arbitrary group shapes. This should reduce redundant global/L1 load instructions versus the naive 7-load stencil, especially for X and common Y-contiguous threadgroup layouts.

```metal
#include <metal_stdlib>
using namespace metal;

kernel void wave_step(device const float *u_prev [[buffer(0)]],
                      device const float *u_curr [[buffer(1)]],
                      device       float *u_next [[buffer(2)]],
                      constant uint      &NX     [[buffer(3)]],
                      constant uint      &NY     [[buffer(4)]],
                      constant uint      &NZ     [[buffer(5)]],
                      constant float     &alpha  [[buffer(6)]],
                      uint3 gid [[thread_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint3 tpg [[threads_per_threadgroup]]) {
    const uint i = gid.x;
    const uint j = gid.y;
    const uint k = gid.z;

    const uint nx = NX;
    const uint ny = NY;
    const uint nz = NZ;

    const bool in_bounds = (i < nx) && (j < ny) && (k < nz);

    const uint stride_y = nx;
    const uint stride_z = nx * ny;
    const uint idx = (k * ny + j) * nx + i;

    float c = 0.0f;
    if (in_bounds) {
        c = u_curr[idx];
    }

    // Keep all lanes participating in the SIMD exchange, including boundary
    // lanes and any padded out-of-grid lanes.  Invalid lanes publish a sentinel
    // tag so they can never be accepted as a stencil neighbor.
    const uint invalid_tag = 0xffffffffu;
    const uint tag = in_bounds ? idx : invalid_tag;

    const ushort self = (ushort)lane;

    const ushort src_xm = (lane > 0u)  ? (ushort)(lane - 1u) : self;
    const ushort src_xp = (lane < 31u) ? (ushort)(lane + 1u) : self;

    const uint  tag_xm = simd_shuffle(tag, src_xm);
    const uint  tag_xp = simd_shuffle(tag, src_xp);
    const float val_xm = simd_shuffle(c,   src_xm);
    const float val_xp = simd_shuffle(c,   src_xp);

    uint  tag_ym = invalid_tag;
    uint  tag_yp = invalid_tag;
    float val_ym = 0.0f;
    float val_yp = 0.0f;

    const uint dy = tpg.x;
    if (dy < 32u) {
        const ushort src_ym = (lane >= dy)        ? (ushort)(lane - dy) : self;
        const ushort src_yp = (lane + dy < 32u)   ? (ushort)(lane + dy) : self;

        tag_ym = simd_shuffle(tag, src_ym);
        tag_yp = simd_shuffle(tag, src_yp);
        val_ym = simd_shuffle(c,   src_ym);
        val_yp = simd_shuffle(c,   src_yp);
    }

    uint  tag_zm = invalid_tag;
    uint  tag_zp = invalid_tag;
    float val_zm = 0.0f;
    float val_zp = 0.0f;

    const uint dz = tpg.x * tpg.y;
    if (dz < 32u) {
        const ushort src_zm = (lane >= dz)        ? (ushort)(lane - dz) : self;
        const ushort src_zp = (lane + dz < 32u)   ? (ushort)(lane + dz) : self;

        tag_zm = simd_shuffle(tag, src_zm);
        tag_zp = simd_shuffle(tag, src_zp);
        val_zm = simd_shuffle(c,   src_zm);
        val_zp = simd_shuffle(c,   src_zp);
    }

    if (!in_bounds) {
        return;
    }

    if (i == 0u || j == 0u || k == 0u ||
        i == nx - 1u || j == ny - 1u || k == nz - 1u) {
        u_next[idx] = c;
        return;
    }

    float xm;
    if (tag_xm == idx - 1u) {
        xm = val_xm;
    } else {
        xm = u_curr[idx - 1u];
    }

    float xp;
    if (tag_xp == idx + 1u) {
        xp = val_xp;
    } else {
        xp = u_curr[idx + 1u];
    }

    float ym;
    if (tag_ym == idx - stride_y) {
        ym = val_ym;
    } else {
        ym = u_curr[idx - stride_y];
    }

    float yp;
    if (tag_yp == idx + stride_y) {
        yp = val_yp;
    } else {
        yp = u_curr[idx + stride_y];
    }

    float zm;
    if (tag_zm == idx - stride_z) {
        zm = val_zm;
    } else {
        zm = u_curr[idx - stride_z];
    }

    float zp;
    if (tag_zp == idx + stride_z) {
        zp = val_zp;
    } else {
        zp = u_curr[idx + stride_z];
    }

    const float lap = xm + xp + ym + yp + zm + zp - 6.0f * c;
    u_next[idx] = 2.0f * c - u_prev[idx] + alpha * lap;
}
```