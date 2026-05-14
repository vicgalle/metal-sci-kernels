## Task: adi3d

3D Locally-One-Dimensional (LOD) ADI for the heat equation. One timestep solves three constant-coefficient tridiagonal systems sequentially along x, then y, then z:
  (I - mu * Dxx) v1      = u^n         (x-sweep)
  (I - mu * Dyy) v2      = v1          (y-sweep)
  (I - mu * Dzz) u^{n+1} = v2          (z-sweep)
where mu = dt/h^2 (host uses mu = 0.5; LOD-ADI is unconditionally stable, no CFL). Each per-line system has constant tridiagonal entries
  -mu * v_{i-1} + (1 + 2 mu) * v_i + -mu * v_{i+1} = rhs_i,  1 <= i <= N-2
with Dirichlet endpoints v_0 = rhs_0, v_{N-1} = rhs_{N-1} (the line's two boundary cells, untouched by the solve).

Cube-face Dirichlet: every cell with i in {0, NX-1} OR j in {0, NY-1} OR k in {0, NZ-1} (any cube face) MUST stay at its initial value across the entire timestep. The harness enforces this convention: per sweep, lines whose two OFF-axis indices both sit strictly interior on the cube get a Thomas solve along the active axis; lines that touch a cube face in their off-axis indices copy u_in -> u_out unchanged. The result is that all six cube faces are preserved through every sub-step.

Storage is row-major float32 of shape (NZ, NY, NX) with i the fast (x) axis, j the middle (y) axis, k the slow (z) axis. Linear index: idx = (k * NY + j) * NX + i. NX, NY, and NZ are independent positive integers and need not be equal. The host calls three separate kernels -- adi_x, adi_y, adi_z -- in that order, ping-ponging two device buffers, with all dispatches sharing one command buffer for accurate end-to-end GPU timing of the n_steps run.

## Required kernel signature(s)

```
kernel void adi_x(device const float *u_in   [[buffer(0)]],
                  device       float *u_out  [[buffer(1)]],
                  constant uint      &NX     [[buffer(2)]],
                  constant uint      &NY     [[buffer(3)]],
                  constant uint      &NZ     [[buffer(4)]],
                  constant float     &mu     [[buffer(5)]],
                  uint2 gid [[thread_position_in_grid]]);
kernel void adi_y(device const float *u_in   [[buffer(0)]],
                  device       float *u_out  [[buffer(1)]],
                  constant uint      &NX     [[buffer(2)]],
                  constant uint      &NY     [[buffer(3)]],
                  constant uint      &NZ     [[buffer(4)]],
                  constant float     &mu     [[buffer(5)]],
                  uint2 gid [[thread_position_in_grid]]);
kernel void adi_z(device const float *u_in   [[buffer(0)]],
                  device       float *u_out  [[buffer(1)]],
                  constant uint      &NX     [[buffer(2)]],
                  constant uint      &NY     [[buffer(3)]],
                  constant uint      &NZ     [[buffer(4)]],
                  constant float     &mu     [[buffer(5)]],
                  uint2 gid [[thread_position_in_grid]]);

Dispatch geometry (host-fixed; identical pattern across the three kernels, with the two off-axis indices on gid.x and gid.y):
  adi_x: threadsPerGrid = (NY, NZ, 1), TG = (32, 1, 1).
         gid.x = j (off-axis y), gid.y = k (off-axis z).
  adi_y: threadsPerGrid = (NX, NZ, 1), TG = (32, 1, 1).
         gid.x = i (off-axis x), gid.y = k (off-axis z).
  adi_z: threadsPerGrid = (NX, NY, 1), TG = (32, 1, 1).
         gid.x = i (off-axis x), gid.y = j (off-axis y).
Convention: one thread owns one full Thomas line along the active axis. Each thread MUST early-exit if its gid is past the corresponding axis length. Boundary lines (those whose off-axis indices touch a cube face) MUST copy u_in -> u_out cell-by-cell.

If you cap the threadgroup with [[max_total_threads_per_threadgroup(W)]], place the attribute on the kernel declaration line itself, and remember the host dispatches TG = (32, 1, 1); a cap below 32 will be rejected. Buffers 0 and 1 are read/write and ping-ponged across timesteps, so do NOT assume u_in and u_out alias fixed addresses. The host calls adi_x -> adi_y -> adi_z back-to-back per timestep, with the output of one sweep being the input of the next; n_steps total timesteps share one command buffer.
```

## Your previous attempt

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
```

Result of previous attempt:
            N64_20: correct, 5.65 ms, 22.3 GB/s (effective, 24 B/cell/step across 3 sweeps) (11.1% of 200 GB/s)
            N96_15: correct, 8.22 ms, 38.7 GB/s (effective, 24 B/cell/step across 3 sweeps) (19.4% of 200 GB/s)
           N128_10: correct, 12.08 ms, 41.7 GB/s (effective, 24 B/cell/step across 3 sweeps) (20.8% of 200 GB/s)
  score (gmean of fraction): 0.1650

## Current best (incumbent)

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

    // Use fast thread-local storage for intermediate Thomas variables
    float dp_arr[MAX_AXIS];

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
                    dp = ((val + mu * bd_hi) - a * dp) / denom;
                } else {
                    float denom = b - a * cprime[i - 1];
                    dp = (val - a * dp) / denom;
                }
                dp_arr[i] = dp;
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
            for (int i_off = int(i_end) - 1; i_off >= 0; --i_off) {
                uint i = i_base + i_off;
                float dpi = dp_arr[i];
                
                if (is_boundary) {
                    tile[tlid][i_off] = dpi;
                } else if (i == NX - 1 || i == NX - 2 || i == 0) {
                    xnext = dpi;
                    tile[tlid][i_off] = dpi;
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

    float dp_arr[MAX_AXIS];
    dp_arr[0] = bd_lo;
    dp_arr[NY - 1] = bd_hi;

    if (NY < 3) {
        for (uint j = 0; j < NY; ++j) {
            u_out[base + j * NX] = dp_arr[j];
        }
        return;
    }

    if (NY == 3) {
        dp_arr[1] = (u_in[base + NX] + mu * bd_lo + mu * bd_hi) / b;
    } else {
        float dp = (u_in[base + NX] + mu * bd_lo) / b;
        dp_arr[1] = dp;

        for (uint j = 2; j + 1 < NY - 1; ++j) {
            float denom = b - a * cprime[j - 1];
            dp = (u_in[base + j * NX] - a * dp) / denom;
            dp_arr[j] = dp;
        }

        uint j = NY - 2;
        float denom = b - a * cprime[j - 1];
        dp = ((u_in[base + j * NX] + mu * bd_hi) - a * dp) / denom;
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

    float dp_arr[MAX_AXIS];
    dp_arr[0] = bd_lo;
    dp_arr[NZ - 1] = bd_hi;

    if (NZ < 3) {
        for (uint k = 0; k < NZ; ++k) {
            u_out[base + k * plane] = dp_arr[k];
        }
        return;
    }

    if (NZ == 3) {
        dp_arr[1] = (u_in[base + plane] + mu * bd_lo + mu * bd_hi) / b;
    } else {
        float dp = (u_in[base + plane] + mu * bd_lo) / b;
        dp_arr[1] = dp;

        for (uint k = 2; k + 1 < NZ - 1; ++k) {
            float denom = b - a * cprime[k - 1];
            dp = (u_in[base + k * plane] - a * dp) / denom;
            dp_arr[k] = dp;
        }

        uint k = NZ - 2;
        float denom = b - a * cprime[k - 1];
        dp = ((u_in[base + k * plane] + mu * bd_hi) - a * dp) / denom;
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

Incumbent result:
            N64_20: correct, 4.02 ms, 31.3 GB/s (effective, 24 B/cell/step across 3 sweeps) (15.7% of 200 GB/s)
            N96_15: correct, 7.70 ms, 41.4 GB/s (effective, 24 B/cell/step across 3 sweeps) (20.7% of 200 GB/s)
           N128_10: correct, 9.72 ms, 51.8 GB/s (effective, 24 B/cell/step across 3 sweeps) (25.9% of 200 GB/s)
  score (gmean of fraction): 0.2032

## History

- iter  0: compile=OK | correct=True | score=0.17196879893902736
- iter  1: compile=OK | correct=True | score=0.19454296300475069
- iter  2: compile=OK | correct=True | score=0.07286105637145143
- iter  3: compile=OK | correct=True | score=0.20316624153339444
- iter  4: compile=OK | correct=True | score=0.16572778468335406
- iter  5: compile=OK | correct=True | score=0.1651520066580215
- iter  6: compile=OK | correct=True | score=0.16503698604064632

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
