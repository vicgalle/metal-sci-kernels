The primary optimization removes massive register spills in `adi_y` and `adi_z` by writing intermediate Thomas variables `d'` directly to `u_out` in the forward sweep and reading them back in the backward sweep, completely avoiding a large dynamically-indexed local array per thread. Since threads in a SIMD group access contiguous memory along `i` for `adi_y` and `adi_z`, these device memory accesses are perfectly coalesced and much faster than spilled local memory. For `adi_x` (which naturally has uncoalesced memory access), we use a `32x33` threadgroup memory tile to chunk the read/compute/write process, maintaining coalesced memory transactions while carrying state across chunks. This bounds threadgroup memory strictly to ~5KB, prevents bank conflicts, and ensures high device bandwidth utilization.

```metal
#include <metal_stdlib>
using namespace metal;

constant constexpr uint MAX_AXIS = 512u;

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
    // Compute Thomas coefficients (shared for all lines in threadgroup)
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
    threadgroup_barrier(mem_flags::mem_threadgroup);

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

    uint num_chunks = (NX + 31) / 32;

    // Forward sweep
    for (uint c = 0; c < num_chunks; ++c) {
        uint i_base = c * 32;
        uint i_load = i_base + tlid;
        bool i_valid = i_load < NX;

        // Cooperative coalesced load from u_in
        for (uint row = 0; row < 32; ++row) {
            uint j_load = j_base + row;
            float val = 0.0f;
            if (i_valid && j_load < NY) {
                val = u_in[base_k + j_load * NX + i_load];
            }
            tile[row][tlid] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute intermediate 'dp' sequentially for the chunk
        if (valid_j) {
            uint i_end = min(32u, NX - i_base);
            for (uint i_off = 0; i_off < i_end; ++i_off) {
                uint i = i_base + i_off;
                float val = tile[tlid][i_off];
                
                if (is_boundary) {
                    dp = val; // Direct copy for boundary lines
                } else if (i == 0) {
                    dp = bd_lo;
                } else if (i == NX - 1) {
                    dp = bd_hi;
                } else if (i == 1) {
                    if (NX == 3) {
                        dp = (val + mu * bd_lo + mu * bd_hi) / b;
                    } else {
                        dp = (val + mu * bd_lo) / b;
                    }
                } else if (i == NX - 2) {
                    float denom = b - a * cprime[i - 1];
                    dp = ((val + mu * bd_hi) - a * dp) / denom;
                } else {
                    float denom = b - a * cprime[i - 1];
                    dp = (val - a * dp) / denom;
                }
                tile[tlid][i_off] = dp;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Cooperative coalesced write intermediate 'dp' to u_out
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

        // Cooperative coalesced load intermediate 'dp' from u_out
        for (uint row = 0; row < 32; ++row) {
            uint j_load = j_base + row;
            float val = 0.0f;
            if (i_valid && j_load < NY) {
                val = u_out[base_k + j_load * NX + i_load];
            }
            tile[row][tlid] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute final value 'xi' in reverse
        if (valid_j) {
            uint i_end = min(32u, NX - i_base);
            for (int i_off = int(i_end) - 1; i_off >= 0; --i_off) {
                uint i = i_base + i_off;
                float dpi = tile[tlid][i_off];
                
                if (is_boundary) {
                    // Do nothing; tile already holds copied u_in value
                } else if (i == NX - 1 || i == NX - 2 || i == 0) {
                    xnext = dpi; // Tile already has the final value
                } else {
                    float xi = dpi - cprime[i] * xnext;
                    tile[tlid][i_off] = xi;
                    xnext = xi;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Cooperative coalesced write final to u_out
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

    // Boundary lines fully copy values over
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

    if (NY < 3) {
        for (uint j = 0; j < NY; ++j) {
            u_out[base + j * NX] = (j == 0) ? bd_lo : bd_hi;
        }
        return;
    }

    float dp = 0.0f;
    if (NY == 3) {
        dp = (u_in[base + NX] + mu * bd_lo + mu * bd_hi) / b;
        u_out[base + NX] = dp;
    } else {
        dp = (u_in[base + NX] + mu * bd_lo) / b;
        u_out[base + NX] = dp; // Persist d' immediately to global mem

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
    float xnext = dp;

    // Solve backward implicitly directly reading the scratch d' from u_out
    for (int jj = NY - 3; jj >= 1; --jj) {
        float dpi = u_out[base + jj * NX];
        float xi  = dpi - cprime[jj] * xnext;
        u_out[base + jj * NX] = xi;
        xnext = xi;
    }
    u_out[base] = bd_lo;
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

    // Boundary lines fully copy values over
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

    if (NZ < 3) {
        for (uint k = 0; k < NZ; ++k) {
            u_out[base + k * plane] = (k == 0) ? bd_lo : bd_hi;
        }
        return;
    }

    float dp = 0.0f;
    if (NZ == 3) {
        dp = (u_in[base + plane] + mu * bd_lo + mu * bd_hi) / b;
        u_out[base + plane] = dp;
    } else {
        dp = (u_in[base + plane] + mu * bd_lo) / b;
        u_out[base + plane] = dp; // Persist d' immediately to global mem

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
    float xnext = dp;

    // Solve backward implicitly directly reading the scratch d' from u_out
    for (int kk = NZ - 3; kk >= 1; --kk) {
        float dpi = u_out[base + kk * plane];
        float xi  = dpi - cprime[kk] * xnext;
        u_out[base + kk * plane] = xi;
        xnext = xi;
    }
    u_out[base] = bd_lo;
}
```