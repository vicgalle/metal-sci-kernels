#include <metal_stdlib>
using namespace metal;

inline uint mix32(uint x) {
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    return x ^ (x >> 16);
}

inline uint ising_rand_bits(uint seed, uint step_idx, uint site_idx) {
    uint x = seed + step_idx * 0x9E3779B9u;
    x = mix32(x);
    x = mix32(x ^ site_idx);
    return x;
}

inline void update_full_rng(device       char  *spins,
                            device const float *p_accept,
                            uint site_idx,
                            int s,
                            int h,
                            uint step_idx,
                            uint seed) {
    const int prod = s * h;
    const uint pidx = uint(prod + 4) >> 1;
    const float pa = p_accept[pidx];

    const uint bits = ising_rand_bits(seed, step_idx, site_idx);
    const float u = float(bits >> 8) * (1.0f / 16777216.0f);

    if (u < pa) {
        spins[site_idx] = (char)(-s);
    }
}

inline void update_skip_certain(device       char  *spins,
                                device const float *p_accept,
                                uint site_idx,
                                int s,
                                int h,
                                uint step_idx,
                                uint seed) {
    const int prod = s * h;

    if (prod <= 0) {
        spins[site_idx] = (char)(-s);
        return;
    }

    const uint pidx = uint(prod + 4) >> 1;
    const float pa = p_accept[pidx];

    if (pa >= 1.0f) {
        spins[site_idx] = (char)(-s);
        return;
    }

    const uint bits = ising_rand_bits(seed, step_idx, site_idx);
    const float u = float(bits >> 8) * (1.0f / 16777216.0f);

    if (u < pa) {
        spins[site_idx] = (char)(-s);
    }
}

kernel void ising_step(device       char  *spins    [[buffer(0)]],
                       device const float *p_accept [[buffer(1)]],
                       constant uint  &NX           [[buffer(2)]],
                       constant uint  &NY           [[buffer(3)]],
                       constant uint  &step_idx     [[buffer(4)]],
                       constant uint  &seed         [[buffer(5)]],
                       uint2 gid [[thread_position_in_grid]],
                       uint2 tid [[thread_position_in_threadgroup]],
                       uint2 tpg [[threads_per_threadgroup]]) {
    const uint nx = NX;
    const uint ny = NY;
    const uint step = step_idx;

    threadgroup char tile[34 * 34];

    const bool use_tile = (nx >= 512u) && (ny >= 512u) &&
                          (tpg.x <= 32u) && (tpg.y <= 32u);

    if (!use_tile) {
        const uint i = gid.x;
        const uint j = gid.y;
        if (i >= nx || j >= ny) return;

        if (((i ^ j ^ step) & 1u) != 0u) return;

        const uint site_idx = j * nx + i;
        const uint total = nx * ny;

        const uint left_idx  = (i == 0u)        ? (site_idx + nx - 1u) : (site_idx - 1u);
        const uint right_idx = (i + 1u == nx)   ? (site_idx + 1u - nx) : (site_idx + 1u);
        const uint up_idx    = (j == 0u)        ? (site_idx + total - nx) : (site_idx - nx);
        const uint down_idx  = (j + 1u == ny)   ? (site_idx + nx - total) : (site_idx + nx);

        const int s = int(spins[site_idx]);
        const int h = int(spins[left_idx]) +
                      int(spins[right_idx]) +
                      int(spins[up_idx]) +
                      int(spins[down_idx]);

        update_full_rng(spins, p_accept, site_idx, s, h, step, seed);
        return;
    }

    const uint group_x0 = gid.x - tid.x;
    const uint group_y0 = gid.y - tid.y;

    const uint rem_x = (group_x0 < nx) ? (nx - group_x0) : 0u;
    const uint rem_y = (group_y0 < ny) ? (ny - group_y0) : 0u;
    const uint valid_w = min(tpg.x, rem_x);
    const uint valid_h = min(tpg.y, rem_y);

    const bool valid = (tid.x < valid_w) && (tid.y < valid_h);
    const uint pitch = tpg.x + 2u;

    uint site_idx = 0u;
    uint i = gid.x;
    uint j = gid.y;

    if (valid) {
        site_idx = j * nx + i;
        const uint total = nx * ny;

        const uint c = (tid.y + 1u) * pitch + (tid.x + 1u);
        tile[c] = spins[site_idx];

        if (tid.x == 0u) {
            const uint left_idx = (i == 0u) ? (site_idx + nx - 1u) : (site_idx - 1u);
            tile[(tid.y + 1u) * pitch] = spins[left_idx];
        }

        if (tid.x == valid_w - 1u) {
            const uint right_idx = (i + 1u == nx) ? (site_idx + 1u - nx) : (site_idx + 1u);
            tile[(tid.y + 1u) * pitch + (valid_w + 1u)] = spins[right_idx];
        }

        if (tid.y == 0u) {
            const uint up_idx = (j == 0u) ? (site_idx + total - nx) : (site_idx - nx);
            tile[tid.x + 1u] = spins[up_idx];
        }

        if (tid.y == valid_h - 1u) {
            const uint down_idx = (j + 1u == ny) ? (site_idx + nx - total) : (site_idx + nx);
            tile[(valid_h + 1u) * pitch + (tid.x + 1u)] = spins[down_idx];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!valid) return;
    if (((i ^ j ^ step) & 1u) != 0u) return;

    const uint c = (tid.y + 1u) * pitch + (tid.x + 1u);

    const int s = int(tile[c]);
    const int h = int(tile[c - 1u]) +
                  int(tile[c + 1u]) +
                  int(tile[c - pitch]) +
                  int(tile[c + pitch]);

    update_skip_certain(spins, p_accept, site_idx, s, h, step, seed);
}