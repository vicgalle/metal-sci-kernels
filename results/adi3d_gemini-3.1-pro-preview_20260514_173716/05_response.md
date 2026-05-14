To significantly improve the memory and instruction throughput of the kernels, two main optimizations are applied:
1. **Division Elimination**: Floating-point division latency (~14 cycles) acts as a severe sequential bottleneck inside the Thomas algorithm loops. We precompute `inv_denom[i]` in thread 0 alongside `cprime[i]`, replacing the division inside the sequential loop with a single FMA and a multiplication (latency ~4 cycles). 
2. **Software Pipelining & Divergence Hoisting**: For `adi_y` and `adi_z`, we explicitly load the next value from global memory (`next_val`) a step ahead. This allows the GPU to overlap the memory latency with the sequential FMA instructions. For `adi_x`, the `is_boundary` check is hoisted outside the chunk loops, preventing warp divergence and keeping active threads uniform during the sequential compute pass.

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
    threadgroup float inv_denom[MAX_AXIS];
    if (tlid == 0 && NX >= 3) {
        float b = 1.0f + 2.0f * mu;
        float c = -mu;
        cprime[1] = c / b;
        inv_denom[1] = 1.0f / b;
        for (uint i = 2; i < NX - 1; ++i) {
            float denom = b + mu * cprime[i - 1]; // a = -mu, so b - a * cprime = b + mu * cprime
            cprime[i] = c / denom;
            inv_denom[i] = 1.0f / denom;
        }
    }

    threadgroup float tile[32][33];
    float dp_arr[MAX_AXIS];

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

    float dp = 0.0f;
    uint num_chunks = (NX + 31) / 32;

    // Forward sweep
    for (uint c = 0; c < num_chunks; ++c) {
        uint i_base = c * 32;
        uint i_load = i_base + tlid;
        bool i_valid = i_load < NX;

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
            if (is_boundary) {
                for (uint i_off = 0; i_off < i_end; ++i_off) {
                    dp_arr[i_base + i_off] = tile[tlid][i_off];
                }
            } else {
                for (uint i_off = 0; i_off < i_end; ++i_off) {
                    uint i = i_base + i_off;
                    float val = tile[tlid][i_off];
                    
                    if (i == 0) {
                        dp = bd_lo;
                    } else if (i == NX - 1) {
                        dp = bd_hi;
                    } else if (i == 1) {
                        float rhs = val + mu * bd_lo;
                        if (NX == 3) rhs += mu * bd_hi;
                        dp = rhs * inv_denom[1];
                    } else if (i == NX - 2) {
                        dp = ((val + mu * bd_hi) + mu * dp) * inv_denom[i];
                    } else {
                        dp = (val + mu * dp) * inv_denom[i];
                    }
                    dp_arr[i] = dp;
                }
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

        if (valid_j) {
            uint i_end = min(32u, NX - i_base);
            if (is_boundary) {
                for (int i_off = int(i_end) - 1; i_off >= 0; --i_off) {
                    tile[tlid][i_off] = dp_arr[i_base + i_off];
                }
            } else {
                for (int i_off = int(i_end) - 1; i_off >= 0; --i_off) {
                    uint i = i_base + i_off;
                    float dpi = dp_arr[i];
                    
                    if (i == NX - 1 || i == NX - 2 || i == 0) {
                        xnext = dpi;
                        tile[tlid][i_off] = dpi;
                    } else {
                        float xi = dpi - cprime[i] * xnext;
                        tile[tlid][i_off] = xi;
                        xnext = xi;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

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
    threadgroup float inv_denom[MAX_AXIS];
    if (tlid == 0 && NY >= 3) {
        float b = 1.0f + 2.0f * mu;
        float c = -mu;
        cprime[1] = c / b;
        inv_denom[1] = 1.0f / b;
        for (uint j = 2; j < NY - 1; ++j) {
            float denom = b + mu * cprime[j - 1];
            cprime[j] = c / denom;
            inv_denom[j] = 1.0f / denom;
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

    float dp_arr[MAX_AXIS];
    dp_arr[0] = bd_lo;
    dp_arr[NY - 1] = bd_hi;

    if (NY < 3) {
        for (uint j = 0; j < NY; ++j) {
            u_out[base + j * NX] = dp_arr[j];
        }
        return;
    }

    float inv_b = inv_denom[1];
    if (NY == 3) {
        dp_arr[1] = (u_in[base + NX] + mu * bd_lo + mu * bd_hi) * inv_b;
    } else {
        float dp = (u_in[base + NX] + mu * bd_lo) * inv_b;
        dp_arr[1] = dp;

        uint j = 2;
        float next_val = u_in[base + j * NX];
        for (; j + 1 < NY - 1; ++j) {
            float val = next_val;
            next_val = u_in[base + (j + 1) * NX];
            dp = (val + mu * dp) * inv_denom[j];
            dp_arr[j] = dp;
        }

        dp = ((next_val + mu * bd_hi) + mu * dp) * inv_denom[j];
        dp_arr[j] = dp;
    }

    u_out[base + (NY - 1) * NX] = bd_hi;
    u_out[base + (NY - 2) * NX] = dp_arr[NY - 2];
    float xnext = dp_arr[NY - 2];

    for (int jj = NY - 3; jj >= 1; --jj) {
        float dpi = dp_arr[jj];
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
    threadgroup float inv_denom[MAX_AXIS];
    if (tlid == 0 && NZ >= 3) {
        float b = 1.0f + 2.0f * mu;
        float c = -mu;
        cprime[1] = c / b;
        inv_denom[1] = 1.0f / b;
        for (uint k = 2; k < NZ - 1; ++k) {
            float denom = b + mu * cprime[k - 1];
            cprime[k] = c / denom;
            inv_denom[k] = 1.0f / denom;
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

    float dp_arr[MAX_AXIS];
    dp_arr[0] = bd_lo;
    dp_arr[NZ - 1] = bd_hi;

    if (NZ < 3) {
        for (uint k = 0; k < NZ; ++k) {
            u_out[base + k * plane] = dp_arr[k];
        }
        return;
    }

    float inv_b = inv_denom[1];
    if (NZ == 3) {
        dp_arr[1] = (u_in[base + plane] + mu * bd_lo + mu * bd_hi) * inv_b;
    } else {
        float dp = (u_in[base + plane] + mu * bd_lo) * inv_b;
        dp_arr[1] = dp;

        uint k = 2;
        float next_val = u_in[base + k * plane];
        for (; k + 1 < NZ - 1; ++k) {
            float val = next_val;
            next_val = u_in[base + (k + 1) * plane];
            dp = (val + mu * dp) * inv_denom[k];
            dp_arr[k] = dp;
        }

        dp = ((next_val + mu * bd_hi) + mu * dp) * inv_denom[k];
        dp_arr[k] = dp;
    }

    u_out[base + (NZ - 1) * plane] = bd_hi;
    u_out[base + (NZ - 2) * plane] = dp_arr[NZ - 2];
    float xnext = dp_arr[NZ - 2];

    for (int kk = NZ - 3; kk >= 1; --kk) {
        float dpi = dp_arr[kk];
        float xi  = dpi - cprime[kk] * xnext;
        u_out[base + kk * plane] = xi;
        xnext = xi;
    }
    u_out[base] = bd_lo;
}
```