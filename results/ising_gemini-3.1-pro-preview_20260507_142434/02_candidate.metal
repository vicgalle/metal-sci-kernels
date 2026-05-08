#include <metal_stdlib>
using namespace metal;

kernel void ising_step(device       char  *spins    [[buffer(0)]],
                       device const float *p_accept [[buffer(1)]],
                       constant uint  &NX           [[buffer(2)]],
                       constant uint  &NY           [[buffer(3)]],
                       constant uint  &step_idx     [[buffer(4)]],
                       constant uint  &seed         [[buffer(5)]],
                       uint2 gid [[thread_position_in_grid]],
                       uint2 tid_in_tg [[thread_position_in_threadgroup]],
                       uint2 tg_pos [[threadgroup_position_in_grid]],
                       uint2 tg_size_vec [[threads_per_threadgroup]]) {
    
    uint tg_w = tg_size_vec.x;
    uint tg_h = tg_size_vec.y;
    uint tg_size = tg_w * tg_h;
    uint tile_w = tg_w + 2;
    uint tile_h = tg_h + 2;
    uint tile_size = tile_w * tile_h;

    // Max threadgroup size is 1024. Max tile size is for 1024x1 -> 1026x3 = 3078.
    threadgroup char tile[3078];

    uint min_x = tg_pos.x * tg_w;
    uint min_y = tg_pos.y * tg_h;

    uint tid = tid_in_tg.y * tg_w + tid_in_tg.x;

    // Cooperatively load the active tile + 1-cell halo into threadgroup memory
    for (uint idx = tid; idx < tile_size; idx += tg_size) {
        uint hy = idx / tile_w;
        uint hx = idx % tile_w;

        int load_x = (int)min_x + (int)hx - 1;
        while (load_x < 0) load_x += (int)NX;
        while (load_x >= (int)NX) load_x -= (int)NX;

        int load_y = (int)min_y + (int)hy - 1;
        while (load_y < 0) load_y += (int)NY;
        while (load_y >= (int)NY) load_y -= (int)NY;

        tile[idx] = spins[load_y * NX + load_x];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint i = gid.x;
    uint j = gid.y;
    if (i >= NX || j >= NY) return;

    // Checkerboard early exit (inactive threads safely stop AFTER the barrier)
    uint color = step_idx & 1u;
    if (((i + j) & 1u) != color) return;

    uint tx = tid_in_tg.x + 1;
    uint ty = tid_in_tg.y + 1;

    // 5 fast reads from threadgroup memory
    int s  = tile[ ty      * tile_w + tx ];
    int sl = tile[ ty      * tile_w + tx - 1 ];
    int sr = tile[ ty      * tile_w + tx + 1 ];
    int sd = tile[(ty - 1) * tile_w + tx ];
    int su = tile[(ty + 1) * tile_w + tx ];

    int h = sl + sr + sd + su;
    int prod = s * h;
    int idx_p = (prod + 4) >> 1;
    float pa = p_accept[idx_p];

    // Uniform base hash state (hoisted)
    uint base_x = seed + step_idx * 0x9E3779B9u;
    base_x = (base_x ^ (base_x >> 16)) * 0x85EBCA6Bu;
    base_x = (base_x ^ (base_x >> 13)) * 0xC2B2AE35u;
    base_x = base_x ^ (base_x >> 16);

    // Site-specific hash state
    uint site_idx = j * NX + i;
    uint x = base_x ^ site_idx;
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    x = x ^ (x >> 16);

    float u = float(x >> 8) * (1.0f / 16777216.0f);

    if (u < pa) {
        spins[site_idx] = (char)(-s);
    }
}