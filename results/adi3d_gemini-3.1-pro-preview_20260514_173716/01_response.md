To significantly improve performance, we address the major bottleneck in `adi_x`: massively uncoalesced memory accesses. Because `adi_x` operates along the fast axis `x` while each thread handles a different `y` (row), threads within a SIMD group access global memory with a stride of `NX`.

We resolve this by using a cooperative 32x32 threadgroup-memory tile. The SIMD group collectively loads a 32x32 block of data perfectly coalesced, transposes it through the `tile` array (padded to 33 to eliminate bank conflicts), and then each thread operates on its row sequentially within the Thomas algorithm. The intermediate `dp` variables and the final backward-substituted `x` values are then written back out fully coalesced.

For `adi_y` and `adi_z`, threads along the warp naturally access contiguous global memory, meaning they are already optimally coalesced. Thus, we simply inline and streamline their Thomas solvers.

```metal
#include <metal_stdlib>
using namespace metal;

constant constexpr uint MAX_AXIS = 256u;

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
    threadgroup float cprime[MAX_AXIS];
    if (tlid == 0 && NX >= 3) {
        float a = -mu;
        float b = 1.0f + 2.0f * mu;
        float c = -mu;
        cprime[1] = c / b;
        for (uint i = 2; i < NX - 1; ++i) {
            cprime[i] = c / (b - a * cprime[i - 1]);
        }
    }

    // 32x33 tile avoids bank conflicts during transpose
    threadgroup float tile[32][33];

    uint k = gid.y;
    if (k >= NZ) return;

    uint j_base = gid.x - tlid;
    uint j = gid.x;
    bool valid_j = (j < NY);
    bool is_boundary = (j == 0 || j == NY - 1 || k == 0 || k == NZ - 1);

    uint plane = NX * NY;
    uint base_k = k * plane;

    float bd_lo = 0.0f;
    float bd_hi = 0.0f;
    if (valid_j) {
        bd_lo = u_in[base_k + j * NX + 0];
        bd_hi = u_in[base_k + j * NX + NX - 1];
    }

    float a = -mu;
    float b = 1.0f + 2.0f * mu;
    float dp = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_chunks = (NX + 31) / 32;

    // Forward sweep
    for (uint c = 0; c < num_chunks; ++c) {
        uint i_base = c * 32;
        uint i_load = i_base + tlid;
        bool i_valid = i_load < NX;

        // Cooperative load
        for (uint row = 0; row < 32; ++row) {
            uint j_load = j_base + row;
            float val = 0.0f;
            if (i_valid && j_load < NY) {
                val = u_in[base_k + j_load * NX + i_load];
            }
            tile[row][tlid] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid_j) {
            uint i_end = min(32u, NX - i_base);
            for (uint i_off = 0; i_off < i_end; ++i_off) {
                uint i = i_base + i_off;
                float val = tile[tlid][i_off];
                
                if (is_boundary) {
                    // Do nothing; will just forward the copy
                } else if (i == 0) {
                    dp = bd_lo;
                    tile[tlid][i_off] = dp;
                } else if (i == NX - 1) {
                    dp = bd_hi;
                    tile[tlid][i_off] = dp;
                } else if (i == 1) {
                    if (NX == 3) {
                        dp = (val + mu * bd_lo + mu * bd_hi) / b;
                    } else {
                        dp = (val + mu * bd_lo) / b;
                    }
                    tile[tlid][i_off] = dp;
                } else if (i == NX - 2) {
                    float denom = b - a * cprime[i - 1];
                    dp = ((val + mu * bd_hi) - a * dp) / denom;
                    tile[tlid][i_off] = dp;
                } else {
                    float denom = b - a * cprime[i - 1];
                    dp = (val - a * dp) / denom;
                    tile[tlid][i_off] = dp;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Cooperative scratch write
        for (uint row = 0; row < 32; ++row) {
            uint j_load = j_base + row;
            if (i_valid && j_load < NY) {
                u_out[base_k + j_load * NX + i_load] = tile[row][tlid];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float xnext = 0.0f;

    // Backward sweep
    for (int c = num_chunks - 1; c >= 0; --c) {
        uint i_base = c * 32;
        uint i_load = i_base + tlid;
        bool i_valid = i_load < NX;

        // Cooperative load
        for (uint row = 0; row < 32; ++row) {
            uint j_load = j_base + row;
            float val = 0.0f;
            if (i_valid && j_load < NY) {
                val = u_out[base_k + j_load * NX + i_load];
            }
            tile[row][tlid] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid_j) {
            uint i_end = min(32u, NX - i_base);
            for (int i_off = int(i_end) - 1; i_off >= 0; --i_off) {
                uint i = i_base + i_off;
                float dpi = tile[tlid][i_off];
                
                if (is_boundary) {
                    // Do nothing
                } else if (i == NX - 1 || i == NX - 2 || i == 0) {
                    xnext = dpi;
                } else {
                    float xi = dpi - cprime[i] * xnext;
                    tile[tlid][i_off] = xi;
                    xnext = xi;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Cooperative final write
        for (uint row = 0; row < 32; ++row) {
            uint j_load = j_base + row;
            if (i_valid && j_load < NY) {
                u_out[base_k + j_load * NX + i_load] = tile[row][tlid];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
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
        for (uint i = 2; i < NY - 1; ++i) {
            cprime[i] = c / (b - a * cprime[i - 1]);
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

    float a = -mu;
    float b = 1.0f + 2.0f * mu;

    float bd_lo = u_in[base];
    float bd_hi = u_in[base + (NY - 1) * NX];

    u_out[base]                 = bd_lo;
    u_out[base + (NY - 1) * NX] = bd_hi;

    if (NY < 3) return;

    if (NY == 3) {
        float dp = (u_in[base + NX] + mu * bd_lo + mu * bd_hi) / b;
        u_out[base + NX] = dp;
        return;
    }

    float dp = (u_in[base + NX] + mu * bd_lo) / b;
    u_out[base + NX] = dp;

    for (uint j = 2; j + 1 < NY - 1; ++j) {
        float denom = b - a * cprime[j - 1];
        dp = (u_in[base + j * NX] - a * dp) / denom;
        u_out[base + j * NX] = dp;
    }

    {
        uint j = NY - 2;
        float denom = b - a * cprime[j - 1];
        dp = ((u_in[base + j * NX] + mu * bd_hi) - a * dp) / denom;
        u_out[base + j * NX] = dp;
    }

    float xnext = dp;
    for (uint jj = NY - 3; jj >= 1; --jj) {
        float dpi = u_out[base + jj * NX];
        float xi  = dpi - cprime[jj] * xnext;
        u_out[base + jj * NX] = xi;
        xnext = xi;
        if (jj == 1) break;
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
        for (uint i = 2; i < NZ - 1; ++i) {
            cprime[i] = c / (b - a * cprime[i - 1]);
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

    float a = -mu;
    float b = 1.0f + 2.0f * mu;

    float bd_lo = u_in[base];
    float bd_hi = u_in[base + (NZ - 1) * plane];

    u_out[base]                    = bd_lo;
    u_out[base + (NZ - 1) * plane] = bd_hi;

    if (NZ < 3) return;

    if (NZ == 3) {
        float dp = (u_in[base + plane] + mu * bd_lo + mu * bd_hi) / b;
        u_out[base + plane] = dp;
        return;
    }

    float dp = (u_in[base + plane] + mu * bd_lo) / b;
    u_out[base + plane] = dp;

    for (uint k = 2; k + 1 < NZ - 1; ++k) {
        float denom = b - a * cprime[k - 1];
        dp = (u_in[base + k * plane] - a * dp) / denom;
        u_out[base + k * plane] = dp;
    }

    {
        uint k = NZ - 2;
        float denom = b - a * cprime[k - 1];
        dp = ((u_in[base + k * plane] + mu * bd_hi) - a * dp) / denom;
        u_out[base + k * plane] = dp;
    }

    float xnext = dp;
    for (uint kk = NZ - 3; kk >= 1; --kk) {
        float dpi = u_out[base + kk * plane];
        float xi  = dpi - cprime[kk] * xnext;
        u_out[base + kk * plane] = xi;
        xnext = xi;
        if (kk == 1) break;
    }
}
```