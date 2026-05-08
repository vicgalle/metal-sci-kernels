#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(1024)]]
kernel void ising_step(device       char  *spins    [[buffer(0)]],
                       device const float *p_accept [[buffer(1)]],
                       constant uint  &NX           [[buffer(2)]],
                       constant uint  &NY           [[buffer(3)]],
                       constant uint  &step_idx     [[buffer(4)]],
                       constant uint  &seed         [[buffer(5)]],
                       uint2 gid [[thread_position_in_grid]],
                       uint2 tid [[thread_position_in_threadgroup]],
                       uint2 ttg [[threads_per_threadgroup]]) 
{
    uint i = gid.x;
    uint j = gid.y;
    uint nx = NX;
    uint ny = NY;

    bool valid = (i < nx && j < ny);
    uint site_idx = j * nx + i;

    // row_stride handles up to max threadgroup limits safely.
    // 3100 ensures enough space for max dimension configurations (e.g. 1x1024 or 32x32).
    uint row_stride = ttg.x + 2u;
    uint local_idx = (tid.y + 1u) * row_stride + (tid.x + 1u);

    threadgroup char tile[3100];

    // Cooperative tile and halo load
    if (valid) {
        uint im = (i == 0u)       ? (nx - 1u) : (i - 1u);
        uint ip = (i == nx - 1u)  ? 0u        : (i + 1u);
        uint jm = (j == 0u)       ? (ny - 1u) : (j - 1u);
        uint jp = (j == ny - 1u)  ? 0u        : (j + 1u);
        
        tile[local_idx] = spins[site_idx];

        if (tid.x == 0u) 
            tile[local_idx - 1u] = spins[j * nx + im];
        if (tid.x == ttg.x - 1u || i == nx - 1u) 
            tile[local_idx + 1u] = spins[j * nx + ip];

        if (tid.y == 0u) 
            tile[local_idx - row_stride] = spins[jm * nx + i];
        if (tid.y == ttg.y - 1u || j == ny - 1u) 
            tile[local_idx + row_stride] = spins[jp * nx + i];
    }

    // Ensure all loads are complete before neighborhood reads
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!valid) return;

    uint step = step_idx;
    uint color = step & 1u;
    if (((i + j) & 1u) != color) return;

    // Read spins natively from fast threadgroup memory
    int s  = tile[local_idx];
    int sl = tile[local_idx - 1u];
    int sr = tile[local_idx + 1u];
    int sd = tile[local_idx - row_stride];
    int su = tile[local_idx + row_stride];

    int h    = sl + sr + sd + su;
    int prod = s * h;
    int idx  = (prod + 4) >> 1;

    // Preload p_accept into registers to bypass L1 gather
    float p0 = p_accept[0];
    float p1 = p_accept[1];
    float p2 = p_accept[2];
    float p3 = p_accept[3];
    float p4 = p_accept[4];

    float pa = p0;
    if (idx == 1) pa = p1;
    else if (idx == 2) pa = p2;
    else if (idx == 3) pa = p3;
    else if (idx == 4) pa = p4;

    // Uniform hash invariant
    uint s_val = seed;
    uint base_x = s_val + step * 0x9E3779B9u;
    base_x = (base_x ^ (base_x >> 16)) * 0x85EBCA6Bu;
    base_x = (base_x ^ (base_x >> 13)) * 0xC2B2AE35u;
    base_x = base_x ^ (base_x >> 16);

    // Site-specific RNG
    uint x = base_x ^ site_idx;
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    x = x ^ (x >> 16);

    float u = float(x >> 8) * (1.0f / 16777216.0f);

    if (u < pa) {
        spins[site_idx] = (char)(-s);
    }
}