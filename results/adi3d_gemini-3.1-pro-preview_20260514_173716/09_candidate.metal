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

    // 32x33 tile avoids bank conflicts
    threadgroup float tile[32][33];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_chunks = (NX + 31) / 32;

    // Forward sweep: compute dp and store into u_out
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
            for (uint i_off = 0; i_off < i_end; ++i_off) {
                uint i = i_base + i_off;
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
                tile[tlid][i_off] = dp;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint row = 0; row < 32; ++row) {
            uint j_store = j_base + row;
            if (i_valid && j_store < NY) {
                u_out[base_k + j_store * NX + i_load] = tile[row][tlid];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    float xnext = 0.0f;

    // Backward sweep: read dp from u_out, compute xi, store back to u_out
    for (int c = num_chunks - 1; c >= 0; --c) {
        uint i_base = c * 32;
        uint i_load = i_base + tlid;
        bool i_valid = i_load < NX;

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
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint row = 0; row < 32; ++row) {
            uint j_store = j_base + row;
            if (i_valid && j_store < NY) {
                u_out[base_k + j_store * NX + i_load] = tile[row][tlid];
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

    if (NY < 3) {
        for (uint j = 0; j < NY; ++j) {
            u_out[base + j * NX] = u_in[base + j * NX];
        }
        return;
    }

    bool is_boundary = (i == 0 || i == NX - 1 || k == 0 || k == NZ - 1);
    if (is_boundary) {
        for (uint j = 0; j < NY; ++j) {
            u_out[base + j * NX] = u_in[base + j * NX];
        }
        return;
    }

    float a = -mu;
    float b = 1.0f + 2.0f * mu;

    float bd_lo = u_in[base];
    float bd_hi = u_in[base + (NY - 1) * NX];

    float current_dp = 0.0f;
    for (uint j = 0; j < NY; ++j) {
        float val = u_in[base + j * NX];
        float dp = 0.0f;
        if (j == 0) {
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
        u_out[base + j * NX] = dp;
    }

    float xnext = 0.0f;
    for (int j = NY - 1; j >= 0; --j) {
        float dpi = u_out[base + j * NX];
        float xj = 0.0f;
        if (j == NY - 1 || j == NY - 2 || j == 0) {
            xj = dpi;
            xnext = dpi;
        } else {
            xj = dpi - cprime[j] * xnext;
            xnext = xj;
        }
        u_out[base + j * NX] = xj;
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

    if (NZ < 3) {
        for (uint k = 0; k < NZ; ++k) {
            u_out[base + k * plane] = u_in[base + k * plane];
        }
        return;
    }

    bool is_boundary = (i == 0 || i == NX - 1 || j == 0 || j == NY - 1);
    if (is_boundary) {
        for (uint k = 0; k < NZ; ++k) {
            u_out[base + k * plane] = u_in[base + k * plane];
        }
        return;
    }

    float a = -mu;
    float b = 1.0f + 2.0f * mu;

    float bd_lo = u_in[base];
    float bd_hi = u_in[base + (NZ - 1) * plane];

    float current_dp = 0.0f;
    for (uint k = 0; k < NZ; ++k) {
        float val = u_in[base + k * plane];
        float dp = 0.0f;
        if (k == 0) {
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
        u_out[base + k * plane] = dp;
    }

    float xnext = 0.0f;
    for (int k = NZ - 1; k >= 0; --k) {
        float dpi = u_out[base + k * plane];
        float xk = 0.0f;
        if (k == NZ - 1 || k == NZ - 2 || k == 0) {
            xk = dpi;
            xnext = dpi;
        } else {
            xk = dpi - cprime[k] * xnext;
            xnext = xk;
        }
        u_out[base + k * plane] = xk;
    }
}