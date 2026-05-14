To eliminate performance bottlenecks, the most critical optimization is avoiding intermediate global/local memory accesses for the Thomas algorithm's state variables (`dp` and `cprime`). In the incumbent, dynamically indexing a thread-local array (`dp_arr[i]`) causes the Metal compiler to spill it from registers to device-backed local memory, resulting in terrible memory overhead. Writing `dp` to `u_out` and reading it back is better, but still doubles global memory traffic (2 reads + 2 writes per sweep). 

This solution forces the compiler to keep `dp_arr` entirely in registers by heavily using `#pragma unroll` over a fixed maximum axis size (`256`), thus statically indexing the arrays. For `adi_x`, we pair this fully unrolled Thomas solver with chunked cooperative loads/stores via a 32x33 `threadgroup` tile to ensure completely coalesced, bank-conflict-free global reads/writes. `adi_y` and `adi_z` natively coalesce and only require the unrolled loops. This cuts the overall memory traffic perfectly in half to exactly 1 read and 1 write per cell per sweep.

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
    if (tlid == 0) {
        float a = -mu;
        float b = 1.0f + 2.0f * mu;
        float c = -mu;
        cprime[1] = c / b;
        for (uint i = 2; i < 255; ++i) {
            cprime[i] = c / (b - a * cprime[i - 1]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

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

    if (NX < 3) {
        if (valid_j) {
            for (uint i = 0; i < NX; ++i) {
                u_out[base_k + j * NX + i] = u_in[base_k + j * NX + i];
            }
        }
        return;
    }

    float bd_lo = 0.0f;
    float bd_hi = 0.0f;
    if (valid_j) {
        bd_lo = u_in[base_k + j * NX + 0];
        bd_hi = u_in[base_k + j * NX + NX - 1];
    }

    float a = -mu;
    float b = 1.0f + 2.0f * mu;
    float current_dp = 0.0f;

    // Use fast thread-local register storage forced by static unrolling
    float dp_arr[MAX_AXIS];

    // Forward sweep
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        uint i_base = c * 32;
        if (i_base < NX) {
            uint i_load = i_base + tlid;
            bool i_valid = i_load < NX;

            // Cooperative load into tile
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
                #pragma unroll
                for (int i_off = 0; i_off < 32; ++i_off) {
                    uint i = i_base + i_off;
                    if (i < NX) {
                        float val = tile[tlid][i_off];
                        float dp = 0.0f;
                        
                        if (is_boundary) {
                            dp = val;
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
                            dp = ((val + mu * bd_hi) - a * current_dp) / denom;
                        } else {
                            float denom = b - a * cprime[i - 1];
                            dp = (val - a * current_dp) / denom;
                        }
                        current_dp = dp;
                        dp_arr[i] = dp;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    float xnext = 0.0f;

    // Backward sweep
    #pragma unroll
    for (int c = 7; c >= 0; --c) {
        uint i_base = c * 32;
        if (i_base < NX) {
            uint i_load = i_base + tlid;
            bool i_valid = i_load < NX;

            if (valid_j) {
                #pragma unroll
                for (int i_off = 31; i_off >= 0; --i_off) {
                    uint i = i_base + i_off;
                    if (i < NX) {
                        float dpi = dp_arr[i];
                        float xi = 0.0f;
                        
                        if (is_boundary) {
                            xi = dpi;
                        } else if (i == NX - 1 || i == NX - 2 || i == 0) {
                            xi = dpi;
                            xnext = dpi;
                        } else {
                            xi = dpi - cprime[i] * xnext;
                            xnext = xi;
                        }
                        tile[tlid][i_off] = xi;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Cooperative final write
            for (uint row = 0; row < 32; ++row) {
                uint j_store = j_base + row;
                if (i_valid && j_store < NY) {
                    u_out[base_k + j_store * NX + i_load] = tile[row][tlid];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
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
    if (tlid == 0) {
        float a = -mu;
        float b = 1.0f + 2.0f * mu;
        float c = -mu;
        cprime[1] = c / b;
        for (uint i = 2; i < 255; ++i) {
            cprime[i] = c / (b - a * cprime[i - 1]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint i = gid.x;
    uint k = gid.y;
    if (i >= NX || k >= NZ) return;

    uint plane = NX * NY;
    uint base  = k * plane + i;

    if (NY < 3) {
        for (uint j = 0; j < NY; ++j) {
            u_out[base + j * NX] = u_in[base + j * NX];
        }
        return;
    }

    bool is_boundary = (i == 0 || i == NX - 1 || k == 0 || k == NZ - 1);

    float bd_lo = u_in[base];
    float bd_hi = u_in[base + (NY - 1) * NX];

    float a = -mu;
    float b = 1.0f + 2.0f * mu;
    float current_dp = 0.0f;
    float dp_arr[MAX_AXIS];

    #pragma unroll
    for (int j = 0; j < 256; ++j) {
        if (j < NY) {
            float val = u_in[base + j * NX];
            float dp = 0.0f;
            
            if (is_boundary) {
                dp = val;
            } else if (j == 0) {
                dp = bd_lo;
            } else if (j == NY - 1) {
                dp = bd_hi;
            } else if (j == 1) {
                if (NY == 3) {
                    dp = (val + mu * bd_lo + mu * bd_hi) / b;
                } else {
                    dp = (val + mu * bd_lo) / b;
                }
            } else if (j == NY - 2) {
                float denom = b - a * cprime[j - 1];
                dp = ((val + mu * bd_hi) - a * current_dp) / denom;
            } else {
                float denom = b - a * cprime[j - 1];
                dp = (val - a * current_dp) / denom;
            }
            current_dp = dp;
            dp_arr[j] = dp;
        }
    }

    float xnext = 0.0f;
    #pragma unroll
    for (int j = 255; j >= 0; --j) {
        if (j < NY) {
            float dpi = dp_arr[j];
            float xj = 0.0f;
            
            if (is_boundary) {
                xj = dpi;
            } else if (j == NY - 1 || j == NY - 2 || j == 0) {
                xj = dpi;
                xnext = dpi;
            } else {
                xj = dpi - cprime[j] * xnext;
                xnext = xj;
            }
            u_out[base + j * NX] = xj;
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
    if (tlid == 0) {
        float a = -mu;
        float b = 1.0f + 2.0f * mu;
        float c = -mu;
        cprime[1] = c / b;
        for (uint k = 2; k < 255; ++k) {
            cprime[k] = c / (b - a * cprime[k - 1]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint i = gid.x;
    uint j = gid.y;
    if (i >= NX || j >= NY) return;

    uint plane = NX * NY;
    uint base  = j * NX + i;

    if (NZ < 3) {
        for (uint k = 0; k < NZ; ++k) {
            u_out[base + k * plane] = u_in[base + k * plane];
        }
        return;
    }

    bool is_boundary = (i == 0 || i == NX - 1 || j == 0 || j == NY - 1);

    float bd_lo = u_in[base];
    float bd_hi = u_in[base + (NZ - 1) * plane];

    float a = -mu;
    float b = 1.0f + 2.0f * mu;
    float current_dp = 0.0f;
    float dp_arr[MAX_AXIS];

    #pragma unroll
    for (int k = 0; k < 256; ++k) {
        if (k < NZ) {
            float val = u_in[base + k * plane];
            float dp = 0.0f;
            
            if (is_boundary) {
                dp = val;
            } else if (k == 0) {
                dp = bd_lo;
            } else if (k == NZ - 1) {
                dp = bd_hi;
            } else if (k == 1) {
                if (NZ == 3) {
                    dp = (val + mu * bd_lo + mu * bd_hi) / b;
                } else {
                    dp = (val + mu * bd_lo) / b;
                }
            } else if (k == NZ - 2) {
                float denom = b - a * cprime[k - 1];
                dp = ((val + mu * bd_hi) - a * current_dp) / denom;
            } else {
                float denom = b - a * cprime[k - 1];
                dp = (val - a * current_dp) / denom;
            }
            current_dp = dp;
            dp_arr[k] = dp;
        }
    }

    float xnext = 0.0f;
    #pragma unroll
    for (int k = 255; k >= 0; --k) {
        if (k < NZ) {
            float dpi = dp_arr[k];
            float xk = 0.0f;
            
            if (is_boundary) {
                xk = dpi;
            } else if (k == NZ - 1 || k == NZ - 2 || k == 0) {
                xk = dpi;
                xnext = dpi;
            } else {
                xk = dpi - cprime[k] * xnext;
                xnext = xk;
            }
            u_out[base + k * plane] = xk;
        }
    }
}
```