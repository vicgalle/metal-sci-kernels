An optimization that dramatically improves performance is eliminating thread-local arrays (which spill to local memory) and optimizing threadgroup memory usage to avoid bank conflicts. 

For `adi_x`, since memory accesses are strided, we load the entire warp's 32 lines cooperatively into a `32x256` threadgroup tile. To avoid bank conflicts entirely and stay exactly within the 32 KB threadgroup memory limit, we address the tile with a skewed layout: `tile[j * 256 + ((i + j) & 255)]`. Since 32 KB is completely consumed by this tile, the Thomas algorithm coefficients (`cprime`) are elegantly distributed across the SIMD group's registers (8 elements per thread) and retrieved via `simd_broadcast` on the fly, eliminating any extra memory overhead.

For `adi_y` and `adi_z`, threads along the active axis already read and write coalesced memory. We can forgo the intermediate tile and completely eliminate the `dp_arr` register arrays by writing the intermediate `dp` variables directly to `u_out` in the forward sweep, and reading them back in the backward sweep. This drastically reduces register pressure, improves occupancy, and lets the L1/L2 caches gracefully handle the intermediate state.

```metal
#include <metal_stdlib>
using namespace metal;

constant constexpr uint MAX_AXIS = 256u;

#define READ_CPRIME(i, my_cp) \
    simd_broadcast( \
        (((i) & 7) == 0 ? my_cp[0] : \
        (((i) & 7) == 1 ? my_cp[1] : \
        (((i) & 7) == 2 ? my_cp[2] : \
        (((i) & 7) == 3 ? my_cp[3] : \
        (((i) & 7) == 4 ? my_cp[4] : \
        (((i) & 7) == 5 ? my_cp[5] : \
        (((i) & 7) == 6 ? my_cp[6] : my_cp[7]))))))), \
        ((i) >> 3))

[[max_total_threads_per_threadgroup(32)]]
kernel void adi_x(device const float *u_in   [[buffer(0)]],
                  device       float *u_out  [[buffer(1)]],
                  constant uint      &NX     [[buffer(2)]],
                  constant uint      &NY     [[buffer(3)]],
                  constant uint      &NZ     [[buffer(4)]],
                  constant float     &mu     [[buffer(5)]],
                  uint2 gid  [[thread_position_in_grid]],
                  uint  tlid [[thread_index_in_threadgroup]])
{
    float my_cp[8];
    float a = -mu;
    float b = 1.0f + 2.0f * mu;
    float c = -mu;
    float current_cp = c / b;
    for (uint idx = 1; idx < NX - 1; ++idx) {
        if ((idx >> 3) == tlid) {
            if ((idx & 7) == 0) my_cp[0] = current_cp;
            else if ((idx & 7) == 1) my_cp[1] = current_cp;
            else if ((idx & 7) == 2) my_cp[2] = current_cp;
            else if ((idx & 7) == 3) my_cp[3] = current_cp;
            else if ((idx & 7) == 4) my_cp[4] = current_cp;
            else if ((idx & 7) == 5) my_cp[5] = current_cp;
            else if ((idx & 7) == 6) my_cp[6] = current_cp;
            else my_cp[7] = current_cp;
        }
        current_cp = c / (b - a * current_cp);
    }

    threadgroup float tile[8192];
#define TILE_IDX(j, i) ((j) * 256 + (((i) + (j)) & 255))

    uint k = gid.y;
    if (k >= NZ) return;

    uint j_base = gid.x - tlid;
    uint plane = NX * NY;
    uint base_k = k * plane;

    for (uint j_local = 0; j_local < 32; ++j_local) {
        uint j_global = j_base + j_local;
        bool valid = (j_global < NY);
        for (uint i_local = tlid; i_local < NX; i_local += 32) {
            float val = 0.0f;
            if (valid) {
                val = u_in[base_k + j_global * NX + i_local];
            }
            tile[TILE_IDX(j_local, i_local)] = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint j = gid.x;
    bool is_boundary = (j == 0 || j == NY - 1 || k == 0 || k == NZ - 1);

    if (j < NY && !is_boundary) {
        float bd_lo = tile[TILE_IDX(tlid, 0)];
        float bd_hi = tile[TILE_IDX(tlid, NX - 1)];

        float dp = 0.0f;

        if (NX == 3) {
            float val = tile[TILE_IDX(tlid, 1)];
            dp = (val + mu * bd_lo + mu * bd_hi) / b;
            tile[TILE_IDX(tlid, 1)] = dp;
        } else if (NX > 3) {
            float val = tile[TILE_IDX(tlid, 1)];
            dp = (val + mu * bd_lo) / b;
            tile[TILE_IDX(tlid, 1)] = dp;

            for (uint i = 2; i < NX - 2; ++i) {
                float denom = b - a * READ_CPRIME(i - 1, my_cp);
                val = tile[TILE_IDX(tlid, i)];
                dp = (val - a * dp) / denom;
                tile[TILE_IDX(tlid, i)] = dp;
            }

            uint i = NX - 2;
            float denom = b - a * READ_CPRIME(i - 1, my_cp);
            val = tile[TILE_IDX(tlid, i)];
            dp = ((val + mu * bd_hi) - a * dp) / denom;
            tile[TILE_IDX(tlid, i)] = dp;
        }

        if (NX >= 3) {
            float xnext = tile[TILE_IDX(tlid, NX - 2)];
            for (int i = NX - 3; i >= 1; --i) {
                float dpi = tile[TILE_IDX(tlid, i)];
                float xi = dpi - READ_CPRIME(i, my_cp) * xnext;
                tile[TILE_IDX(tlid, i)] = xi;
                xnext = xi;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j_local = 0; j_local < 32; ++j_local) {
        uint j_global = j_base + j_local;
        bool valid = (j_global < NY);
        for (uint i_local = tlid; i_local < NX; i_local += 32) {
            if (valid) {
                u_out[base_k + j_global * NX + i_local] = tile[TILE_IDX(j_local, i_local)];
            }
        }
    }
}

[[max_total_threads_per_threadgroup(32)]]
kernel void adi_y(device const float *u_in   [[buffer(0)]],
                  device       float *u_out  [[buffer(1)]],
                  constant uint      &NX     [[buffer(2)]],
                  constant uint      &NY     [[buffer(3)]],
                  constant uint      &NZ     [[buffer(4)]],
                  constant float     &mu     [[buffer(5)]],
                  uint2 gid  [[thread_position_in_grid]],
                  uint  tlid [[thread_index_in_threadgroup]])
{
    threadgroup float cprime[MAX_AXIS];
    if (tlid == 0 && NY >= 3) {
        float a = -mu;
        float b = 1.0f + 2.0f * mu;
        float c = -mu;
        cprime[1] = c / b;
        for (uint j = 2; j < NY - 1; ++j) {
            cprime[j] = c / (b - a * cprime[j - 1]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint i = gid.x;
    uint k = gid.y;
    if (i >= NX || k >= NZ) return;

    uint plane = NX * NY;
    uint base  = k * plane + i;

    if (i == 0 || i == NX - 1 || k == 0 || k == NZ - 1) {
        for (uint j = 0; j < NY; ++j) {
            u_out[base + j * NX] = u_in[base + j * NX];
        }
        return;
    }

    float bd_lo = u_in[base];
    float bd_hi = u_in[base + (NY - 1) * NX];

    if (NY < 3) {
        u_out[base] = bd_lo;
        if (NY == 2) u_out[base + NX] = bd_hi;
        return;
    }

    float a = -mu;
    float b = 1.0f + 2.0f * mu;

    if (NY == 3) {
        float dp1 = (u_in[base + NX] + mu * bd_lo + mu * bd_hi) / b;
        u_out[base + NX] = dp1;
    } else {
        float dp = (u_in[base + NX] + mu * bd_lo) / b;
        u_out[base + NX] = dp;

        for (uint j = 2; j + 1 < NY - 1; ++j) {
            float denom = b - a * cprime[j - 1];
            dp = (u_in[base + j * NX] - a * dp) / denom;
            u_out[base + j * NX] = dp;
        }

        uint j = NY - 2;
        float denom = b - a * cprime[j - 1];
        dp = ((u_in[base + j * NX] + mu * bd_hi) - a * dp) / denom;
        u_out[base + j * NX] = dp;
    }

    u_out[base + (NY - 1) * NX] = bd_hi;
    u_out[base] = bd_lo;

    if (NY >= 3) {
        float xnext = u_out[base + (NY - 2) * NX];
        for (int jj = NY - 3; jj >= 1; --jj) {
            float dpi = u_out[base + jj * NX];
            float xi  = dpi - cprime[jj] * xnext;
            u_out[base + jj * NX] = xi;
            xnext = xi;
        }
    }
}

[[max_total_threads_per_threadgroup(32)]]
kernel void adi_z(device const float *u_in   [[buffer(0)]],
                  device       float *u_out  [[buffer(1)]],
                  constant uint      &NX     [[buffer(2)]],
                  constant uint      &NY     [[buffer(3)]],
                  constant uint      &NZ     [[buffer(4)]],
                  constant float     &mu     [[buffer(5)]],
                  uint2 gid  [[thread_position_in_grid]],
                  uint  tlid [[thread_index_in_threadgroup]])
{
    threadgroup float cprime[MAX_AXIS];
    if (tlid == 0 && NZ >= 3) {
        float a = -mu;
        float b = 1.0f + 2.0f * mu;
        float c = -mu;
        cprime[1] = c / b;
        for (uint k = 2; k < NZ - 1; ++k) {
            cprime[k] = c / (b - a * cprime[k - 1]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint i = gid.x;
    uint j = gid.y;
    if (i >= NX || j >= NY) return;

    uint plane = NX * NY;
    uint base  = j * NX + i;

    if (i == 0 || i == NX - 1 || j == 0 || j == NY - 1) {
        for (uint k = 0; k < NZ; ++k) {
            u_out[base + k * plane] = u_in[base + k * plane];
        }
        return;
    }

    float bd_lo = u_in[base];
    float bd_hi = u_in[base + (NZ - 1) * plane];

    if (NZ < 3) {
        u_out[base] = bd_lo;
        if (NZ == 2) u_out[base + plane] = bd_hi;
        return;
    }

    float a = -mu;
    float b = 1.0f + 2.0f * mu;

    if (NZ == 3) {
        float dp1 = (u_in[base + plane] + mu * bd_lo + mu * bd_hi) / b;
        u_out[base + plane] = dp1;
    } else {
        float dp = (u_in[base + plane] + mu * bd_lo) / b;
        u_out[base + plane] = dp;

        for (uint k = 2; k + 1 < NZ - 1; ++k) {
            float denom = b - a * cprime[k - 1];
            dp = (u_in[base + k * plane] - a * dp) / denom;
            u_out[base + k * plane] = dp;
        }

        uint k = NZ - 2;
        float denom = b - a * cprime[k - 1];
        dp = ((u_in[base + k * plane] + mu * bd_hi) - a * dp) / denom;
        u_out[base + k * plane] = dp;
    }

    u_out[base + (NZ - 1) * plane] = bd_hi;
    u_out[base] = bd_lo;

    if (NZ >= 3) {
        float xnext = u_out[base + (NZ - 2) * plane];
        for (int kk = NZ - 3; kk >= 1; --kk) {
            float dpi = u_out[base + kk * plane];
            float xi  = dpi - cprime[kk] * xnext;
            u_out[base + kk * plane] = xi;
            xnext = xi;
        }
    }
}
```