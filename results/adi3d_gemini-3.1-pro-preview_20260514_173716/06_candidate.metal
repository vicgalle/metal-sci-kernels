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
    threadgroup float2 coeffs[MAX_AXIS];
    if (tlid == 0 && NX >= 3) {
        float b = 1.0f + 2.0f * mu;
        float c = -mu;
        float cp = c / b;
        coeffs[1] = float2(cp, 1.0f / b);
        for (uint i = 2; i < NX - 1; ++i) {
            float denom = b + mu * cp;
            cp = c / denom;
            coeffs[i] = float2(cp, 1.0f / denom);
        }
    }

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

    float dp = 0.0f;
    float dp_arr[MAX_AXIS];
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
                        float2 cd = coeffs[1];
                        if (NX == 3) {
                            dp = (val + mu * bd_lo + mu * bd_hi) * cd.y;
                        } else {
                            dp = (val + mu * bd_lo) * cd.y;
                        }
                    } else if (i == NX - 2) {
                        float2 cd = coeffs[i];
                        dp = (val + mu * bd_hi + mu * dp) * cd.y;
                    } else {
                        float2 cd = coeffs[i];
                        dp = (val + mu * dp) * cd.y;
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
                        float2 cd = coeffs[i];
                        float xi = dpi - cd.x * xnext;
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
    threadgroup float2 coeffs[MAX_AXIS];
    if (tlid == 0 && NY >= 3) {
        float b = 1.0f + 2.0f * mu;
        float c = -mu;
        float cp = c / b;
        coeffs[1] = float2(cp, 1.0f / b);
        for (uint i = 2; i < NY - 1; ++i) {
            float denom = b + mu * cp;
            cp = c / denom;
            coeffs[i] = float2(cp, 1.0f / denom);
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

    if (NY < 3) {
        u_out[base] = bd_lo;
        if (NY == 2) u_out[base + NX] = bd_hi;
        return;
    }

    if (NY == 3) {
        dp_arr[1] = (u_in[base + NX] + mu * bd_lo + mu * bd_hi) * coeffs[1].y;
    } else {
        float dp = (u_in[base + NX] + mu * bd_lo) * coeffs[1].y;
        dp_arr[1] = dp;

        float next_val = u_in[base + 2 * NX];
        for (uint j = 2; j + 1 < NY - 1; ++j) {
            float val = next_val;
            next_val = u_in[base + (j + 1) * NX];
            dp = (val + mu * dp) * coeffs[j].y;
            dp_arr[j] = dp;
        }

        dp = (next_val + mu * bd_hi + mu * dp) * coeffs[NY - 2].y;
        dp_arr[NY - 2] = dp;
    }

    u_out[base + (NY - 1) * NX] = bd_hi;
    u_out[base + (NY - 2) * NX] = dp_arr[NY - 2];
    float xnext = dp_arr[NY - 2];

    for (int jj = NY - 3; jj >= 1; --jj) {
        float dpi = dp_arr[jj];
        float xi  = dpi - coeffs[jj].x * xnext;
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
    threadgroup float2 coeffs[MAX_AXIS];
    if (tlid == 0 && NZ >= 3) {
        float b = 1.0f + 2.0f * mu;
        float c = -mu;
        float cp = c / b;
        coeffs[1] = float2(cp, 1.0f / b);
        for (uint i = 2; i < NZ - 1; ++i) {
            float denom = b + mu * cp;
            cp = c / denom;
            coeffs[i] = float2(cp, 1.0f / denom);
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

    if (NZ < 3) {
        u_out[base] = bd_lo;
        if (NZ == 2) u_out[base + plane] = bd_hi;
        return;
    }

    if (NZ == 3) {
        dp_arr[1] = (u_in[base + plane] + mu * bd_lo + mu * bd_hi) * coeffs[1].y;
    } else {
        float dp = (u_in[base + plane] + mu * bd_lo) * coeffs[1].y;
        dp_arr[1] = dp;

        float next_val = u_in[base + 2 * plane];
        for (uint k = 2; k + 1 < NZ - 1; ++k) {
            float val = next_val;
            next_val = u_in[base + (k + 1) * plane];
            dp = (val + mu * dp) * coeffs[k].y;
            dp_arr[k] = dp;
        }

        dp = (next_val + mu * bd_hi + mu * dp) * coeffs[NZ - 2].y;
        dp_arr[NZ - 2] = dp;
    }

    u_out[base + (NZ - 1) * plane] = bd_hi;
    u_out[base + (NZ - 2) * plane] = dp_arr[NZ - 2];
    float xnext = dp_arr[NZ - 2];

    for (int kk = NZ - 3; kk >= 1; --kk) {
        float dpi = dp_arr[kk];
        float xi  = dpi - coeffs[kk].x * xnext;
        u_out[base + kk * plane] = xi;
        xnext = xi;
    }
    u_out[base] = bd_lo;
}