## Task: lj

Lennard-Jones molecular dynamics with a cell-list spatial hash. Cubic periodic box of side L; cutoff rcut = 2.5 (sigma = epsilon = mass = 1).

Per timestep, three kernels are dispatched in this fixed order:
  1) lj_clear_cells: zero the per-cell occupancy counter (M^3 threads).
  2) lj_build_cells: each particle thread computes its cell index (after wrapping its position into [0, L)) and atomically appends itself to that cell (N threads).
  3) lj_step: each particle thread iterates the 27 neighbour cells (its own cell + 3^3 - 1 face/edge/corner neighbours, with periodic wrap), reads each occupant from cell_list, and sums the Lennard-Jones force from those within rcut. It then takes one symplectic-Euler step:  v_new = v + a*dt; r_new = r + v_new*dt (N threads).

Cell layout: M cells per side; cell index = (cz*M + cy)*M + cx; cell_size = L/M is guaranteed >= rcut so 27 neighbour cells cover all interactions. cell_count[M^3] holds the per-cell occupancy, cell_list[M^3 * MAX_PER_CELL] holds the particle indices, with row-major slot order. MAX_PER_CELL = 64 is generous for the supplied initial states; particles exceeding this cap are silently dropped (the seed tolerates this since the well-conditioned initial state never overflows, and a candidate may rely on the same invariant).

Lennard-Jones force on i from j (sigma = epsilon = 1):
  d = (r_j - r_i), minimum-image:  d -= L * round(d / L)
  r2 = dot(d, d); skip if r2 >= rcut^2 or r2 ~= 0
  inv_r2 = 1/r2; inv_r6 = inv_r2^3; inv_r12 = inv_r6^2
  F_on_i = -24 * (2*inv_r12 - inv_r6) * inv_r2 * d
  a_i = sum of F_on_i over all j within cutoff (mass = 1).

Positions/velocities are stored as float4 with .xyz holding the data and .w padding (matches the nbody task's layout). The host ping-pongs (pos_in, pos_out) and (vel_in, vel_out) buffer pairs each step; cell_count and cell_list are scratch buffers reused every step (cleared by lj_clear_cells).

## Required kernel signature(s)

```
kernel void lj_clear_cells(
    device atomic_uint *cell_count [[buffer(0)]],
    constant uint      &M3         [[buffer(1)]],
    uint gid [[thread_position_in_grid]]);

kernel void lj_build_cells(
    device const float4 *pos          [[buffer(0)]],
    device atomic_uint  *cell_count   [[buffer(1)]],
    device       uint   *cell_list    [[buffer(2)]],
    constant uint        &N           [[buffer(3)]],
    constant uint        &M           [[buffer(4)]],
    constant float       &L           [[buffer(5)]],
    constant uint        &MAX_PER_CELL[[buffer(6)]],
    uint i [[thread_position_in_grid]]);

kernel void lj_step(
    device const float4 *pos_in       [[buffer(0)]],
    device       float4 *pos_out      [[buffer(1)]],
    device const float4 *vel_in       [[buffer(2)]],
    device       float4 *vel_out      [[buffer(3)]],
    device const uint   *cell_count   [[buffer(4)]],
    device const uint   *cell_list    [[buffer(5)]],
    constant uint        &N           [[buffer(6)]],
    constant uint        &M           [[buffer(7)]],
    constant float       &L           [[buffer(8)]],
    constant float       &rcut2       [[buffer(9)]],
    constant float       &dt          [[buffer(10)]],
    constant uint        &MAX_PER_CELL[[buffer(11)]],
    uint i [[thread_position_in_grid]]);

All three kernels are dispatched 1-D, one thread per element. lj_clear_cells: M^3 threads (gid >= M3 early-exits). lj_build_cells / lj_step: N threads (i >= N early-exits). Each thread MUST handle exactly one element; the host will not shrink the dispatch if you process multiple elements per thread. All buffers use MTLResourceStorageModeShared (Apple Silicon unified memory). cell_count is read via atomics from lj_clear_cells / lj_build_cells and as a plain uint* in lj_step (no atomicity required for the read-only pass).
```

## Your previous attempt

```metal
#include <metal_stdlib>
using namespace metal;

kernel void lj_clear_cells(
    device atomic_uint *cell_count [[buffer(0)]],
    constant uint      &M3         [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) 
{
    if (gid >= M3) return;
    atomic_store_explicit(&cell_count[gid], 0u, memory_order_relaxed);
}

kernel void lj_build_cells(
    device const float4 *pos          [[buffer(0)]],
    device atomic_uint  *cell_count   [[buffer(1)]],
    device       uint   *cell_list    [[buffer(2)]],
    constant uint        &N           [[buffer(3)]],
    constant uint        &M           [[buffer(4)]],
    constant float       &L           [[buffer(5)]],
    constant uint        &MAX_PER_CELL[[buffer(6)]],
    uint i [[thread_position_in_grid]]) 
{
    if (i >= N) return;
    float3 r = pos[i].xyz;
    float inv_L = 1.0f / L;
    r -= L * floor(r * inv_L);
    
    float inv_cell_size = float(M) * inv_L;
    uint cx = min(uint(r.x * inv_cell_size), M - 1u);
    uint cy = min(uint(r.y * inv_cell_size), M - 1u);
    uint cz = min(uint(r.z * inv_cell_size), M - 1u);
    
    uint cell = (cz * M + cy) * M + cx;
    uint max_per_cell = MAX_PER_CELL;
    uint slot = atomic_fetch_add_explicit(&cell_count[cell], 1u, memory_order_relaxed);
    if (slot < max_per_cell) {
        cell_list[cell * max_per_cell + slot] = i;
    }
}

kernel void lj_step(
    device const float4 *pos_in       [[buffer(0)]],
    device       float4 *pos_out      [[buffer(1)]],
    device const float4 *vel_in       [[buffer(2)]],
    device       float4 *vel_out      [[buffer(3)]],
    device const uint   *cell_count   [[buffer(4)]],
    device const uint   *cell_list    [[buffer(5)]],
    constant uint        &N           [[buffer(6)]],
    constant uint        &M           [[buffer(7)]],
    constant float       &L           [[buffer(8)]],
    constant float       &rcut2       [[buffer(9)]],
    constant float       &dt          [[buffer(10)]],
    constant uint        &MAX_PER_CELL[[buffer(11)]],
    uint i [[thread_position_in_grid]]) 
{
    if (i >= N) return;
    
    float4 pos_i = pos_in[i];
    float4 vel_i = vel_in[i];
    float3 ri = pos_i.xyz;
    float3 vi = vel_i.xyz;
    
    float inv_L = 1.0f / L;
    uint M_u = M;
    float inv_cell_size = float(M_u) * inv_L;
    
    float3 ri_w = ri - L * floor(ri * inv_L);
    uint cx = min(uint(ri_w.x * inv_cell_size), M_u - 1u);
    uint cy = min(uint(ri_w.y * inv_cell_size), M_u - 1u);
    uint cz = min(uint(ri_w.z * inv_cell_size), M_u - 1u);

    float cell_size = L / float(M_u);
    float rx = ri_w.x - float(cx) * cell_size;
    float ry = ri_w.y - float(cy) * cell_size;
    float rz = ri_w.z - float(cz) * cell_size;

    float dx_sq[3] = { rx * rx, 0.0f, (cell_size - rx) * (cell_size - rx) };
    float dy_sq[3] = { ry * ry, 0.0f, (cell_size - ry) * (cell_size - ry) };
    float dz_sq[3] = { rz * rz, 0.0f, (cell_size - rz) * (cell_size - rz) };

    uint cx_n[3] = { cx == 0 ? M_u - 1 : cx - 1, cx, cx == M_u - 1 ? 0 : cx + 1 };
    uint cy_n[3] = { cy == 0 ? M_u - 1 : cy - 1, cy, cy == M_u - 1 ? 0 : cy + 1 };
    uint cz_n[3] = { cz == 0 ? M_u - 1 : cz - 1, cz, cz == M_u - 1 ? 0 : cz + 1 };

    float3 a = float3(0.0f);
    
    uint max_per_cell = MAX_PER_CELL;
    uint chunks_per_cell = max_per_cell / 4;
    device const uint4* cell_list_u4 = (device const uint4*)cell_list;

    #pragma unroll
    for (int z = 0; z < 3; ++z) {
        float z2 = dz_sq[z];
        uint cz_idx = cz_n[z] * M_u * M_u;
        
        #pragma unroll
        for (int y = 0; y < 3; ++y) {
            float zy2 = z2 + dy_sq[y];
            if (zy2 >= rcut2) continue;
            uint cy_idx = cz_idx + cy_n[y] * M_u;
            
            #pragma unroll
            for (int x = 0; x < 3; ++x) {
                float zyx2 = zy2 + dx_sq[x];
                if (zyx2 >= rcut2) continue;
                
                uint nc = cy_idx + cx_n[x];
                uint cnt = min(cell_count[nc], max_per_cell);
                if (cnt == 0) continue;
                
                uint cell_chunk_base = nc * chunks_per_cell;
                uint num_chunks = (cnt + 3) / 4;
                
                for (uint k = 0; k < num_chunks; ++k) {
                    uint4 j4 = cell_list_u4[cell_chunk_base + k];
                    uint base_p = k * 4;
                    
                    uint4 valid_mask = uint4(base_p) + uint4(0, 1, 2, 3) < uint4(cnt);
                    uint4 j_safe = select(uint4(i), j4, valid_mask);
                    
                    float4 p0 = pos_in[j_safe.x];
                    float4 p1 = pos_in[j_safe.y];
                    float4 p2 = pos_in[j_safe.z];
                    float4 p3 = pos_in[j_safe.w];
                    
                    float4 dx = float4(p0.x, p1.x, p2.x, p3.x) - ri.x;
                    float4 dy = float4(p0.y, p1.y, p2.y, p3.y) - ri.y;
                    float4 dz = float4(p0.z, p1.z, p2.z, p3.z) - ri.z;
                    
                    dx -= L * rint(dx * inv_L);
                    dy -= L * rint(dy * inv_L);
                    dz -= L * rint(dz * inv_L);
                    
                    float4 r2 = dx*dx + dy*dy + dz*dz;
                    
                    bool4 mask = (r2 < rcut2) && (r2 > 1e-6f);
                    r2 = select(float4(1.0f), r2, mask);
                    
                    float4 ir2 = 1.0f / r2;
                    float4 ir6 = ir2 * ir2 * ir2;
                    float4 f_mag = select(float4(0.0f), ir6 * ir2 * (24.0f - 48.0f * ir6), mask);
                    
                    a.x += dot(f_mag, dx);
                    a.y += dot(f_mag, dy);
                    a.z += dot(f_mag, dz);
                }
            }
        }
    }

    float3 v_new = vi + a * dt;
    float3 r_new = ri + v_new * dt;
    pos_out[i] = float4(r_new, pos_i.w);
    vel_out[i] = float4(v_new, vel_i.w);
}
```

Result of previous attempt:
  COMPILE FAILED: Error Domain=MTLLibraryErrorDomain Code=3 "program_source:118:27: error: cannot initialize a variable of type 'uint4' (vector of 4 'unsigned int' values) with an rvalue of type 'bool __attribute__((ext_vector_type(4)))' (vector of 4 'bool' values)
                    uint4 valid_mask = uint4(base_p) + uint4(0, 1, 2, 3) < uint4(cnt);
                          ^            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
program_source:119:36: error: no matching function for call to 'select'
                    uint4 j_safe = select(uint4(i), j4, valid_mask);
                                   ^~~~~~
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1032:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::bool4' (aka 'bool4') for 3rd argument
METAL_FUNC uint4 select(uint4 x, uint4 y, bool4 c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:852:17: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'bool' for 1st argument
METAL_FUNC bool select(bool x, bool y, bool c)
                ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:856:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::bool2' (aka 'bool2') for 1st argument
METAL_FUNC bool2 select(bool2 x, bool2 y, bool2 c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:860:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::bool3' (aka 'bool3') for 1st argument
METAL_FUNC bool3 select(bool3 x, bool3 y, bool3 c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:864:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::bool4' (aka 'bool4') for 1st argument
METAL_FUNC bool4 select(bool4 x, bool4 y, bool4 c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:880:17: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'char' for 1st argument
METAL_FUNC char select(char x, char y, bool c)
                ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:884:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::char2' (aka 'char2') for 1st argument
METAL_FUNC char2 select(char2 x, char2 y, bool2 c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:888:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::char3' (aka 'char3') for 1st argument
METAL_FUNC char3 select(char3 x, char3 y, bool3 c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:892:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::char4' (aka 'char4') for 1st argument
METAL_FUNC char4 select(char4 x, char4 y, bool4 c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:908:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::uchar' (aka 'unsigned char') for 1st argument
METAL_FUNC uchar select(uchar x, uchar y, bool c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:912:19: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::uchar2' (aka 'uchar2') for 1st argument
METAL_FUNC uchar2 select(uchar2 x, uchar2 y, bool2 c)
                  ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:916:19: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::uchar3' (aka 'uchar3') for 1st argument
METAL_FUNC uchar3 select(uchar3 x, uchar3 y, bool3 c)
                  ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:920:19: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::uchar4' (aka 'uchar4') for 1st argument
METAL_FUNC uchar4 select(uchar4 x, uchar4 y, bool4 c)
                  ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:936:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'short' for 1st argument
METAL_FUNC short select(short x, short y, bool c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:940:19: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::short2' (aka 'short2') for 1st argument
METAL_FUNC short2 select(short2 x, short2 y, bool2 c)
                  ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:944:19: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::short3' (aka 'short3') for 1st argument
METAL_FUNC short3 select(short3 x, short3 y, bool3 c)
                  ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:948:19: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::short4' (aka 'short4') for 1st argument
METAL_FUNC short4 select(short4 x, short4 y, bool4 c)
                  ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:964:19: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::ushort' (aka 'unsigned short') for 1st argument
METAL_FUNC ushort select(ushort x, ushort y, bool c)
                  ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:968:20: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::ushort2' (aka 'ushort2') for 1st argument
METAL_FUNC ushort2 select(ushort2 x, ushort2 y, bool2 c)
                   ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:972:20: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::ushort3' (aka 'ushort3') for 1st argument
METAL_FUNC ushort3 select(ushort3 x, ushort3 y, bool3 c)
                   ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:976:20: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::ushort4' (aka 'ushort4') for 1st argument
METAL_FUNC ushort4 select(ushort4 x, ushort4 y, bool4 c)
                   ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:992:16: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'int' for 1st argument
METAL_FUNC int select(int x, int y, bool c)
               ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:996:17: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::int2' (aka 'int2') for 1st argument
METAL_FUNC int2 select(int2 x, int2 y, bool2 c)
                ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1000:17: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::int3' (aka 'int3') for 1st argument
METAL_FUNC int3 select(int3 x, int3 y, bool3 c)
                ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1004:17: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::int4' (aka 'int4') for 1st argument
METAL_FUNC int4 select(int4 x, int4 y, bool4 c)
                ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1020:17: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::uint' (aka 'unsigned int') for 1st argument
METAL_FUNC uint select(uint x, uint y, bool c)
                ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1024:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::uint2' (aka 'uint2') for 1st argument
METAL_FUNC uint2 select(uint2 x, uint2 y, bool2 c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1028:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::uint3' (aka 'uint3') for 1st argument
METAL_FUNC uint3 select(uint3 x, uint3 y, bool3 c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1049:17: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'long' for 1st argument
METAL_FUNC long select(long x, long y, bool c)
                ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1055:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::long2' (aka 'long2') for 1st argument
METAL_FUNC long2 select(long2 x, long2 y, bool2 c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1061:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::long3' (aka 'long3') for 1st argument
METAL_FUNC long3 select(long3 x, long3 y, bool3 c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1067:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::long4' (aka 'long4') for 1st argument
METAL_FUNC long4 select(long4 x, long4 y, bool4 c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1089:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::ulong' (aka 'unsigned long') for 1st argument
METAL_FUNC ulong select(ulong x, ulong y, bool c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1095:19: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::ulong2' (aka 'ulong2') for 1st argument
METAL_FUNC ulong2 select(ulong2 x, ulong2 y, bool2 c)
                  ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1101:19: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::ulong3' (aka 'ulong3') for 1st argument
METAL_FUNC ulong3 select(ulong3 x, ulong3 y, bool3 c)
                  ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1107:19: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::ulong4' (aka 'ulong4') for 1st argument
METAL_FUNC ulong4 select(ulong4 x, ulong4 y, bool4 c)
                  ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1168:17: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'half' for 1st argument
METAL_FUNC half select(half x, half y, bool c)
                ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1172:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::half2' (aka 'half2') for 1st argument
METAL_FUNC half2 select(half2 x, half2 y, bool2 c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1176:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::half3' (aka 'half3') for 1st argument
METAL_FUNC half3 select(half3 x, half3 y, bool3 c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1180:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::half4' (aka 'half4') for 1st argument
METAL_FUNC half4 select(half4 x, half4 y, bool4 c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1196:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'float' for 1st argument
METAL_FUNC float select(float x, float y, bool c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1200:19: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::float2' (aka 'float2') for 1st argument
METAL_FUNC float2 select(float2 x, float2 y, bool2 c)
                  ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1204:19: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::float3' (aka 'float3') for 1st argument
METAL_FUNC float3 select(float3 x, float3 y, bool3 c)
                  ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1208:19: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::float4' (aka 'float4') for 1st argument
METAL_FUNC float4 select(float4 x, float4 y, bool4 c)
                  ^
" UserInfo={NSLocalizedDescription=program_source:118:27: error: cannot initialize a variable of type 'uint4' (vector of 4 'unsigned int' values) with an rvalue of type 'bool __attribute__((ext_vector_type(4)))' (vector of 4 'bool' values)
                    uint4 valid_mask = uint4(base_p) + uint4(0, 1, 2, 3) < uint4(cnt);
                          ^            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
program_source:119:36: error: no matching function for call to 'select'
                    uint4 j_safe = select(uint4(i), j4, valid_mask);
                                   ^~~~~~
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1032:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::bool4' (aka 'bool4') for 3rd argument
METAL_FUNC uint4 select(uint4 x, uint4 y, bool4 c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:852:17: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'bool' for 1st argument
METAL_FUNC bool select(bool x, bool y, bool c)
                ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:856:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::bool2' (aka 'bool2') for 1st argument
METAL_FUNC bool2 select(bool2 x, bool2 y, bool2 c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:860:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::bool3' (aka 'bool3') for 1st argument
METAL_FUNC bool3 select(bool3 x, bool3 y, bool3 c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:864:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::bool4' (aka 'bool4') for 1st argument
METAL_FUNC bool4 select(bool4 x, bool4 y, bool4 c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:880:17: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'char' for 1st argument
METAL_FUNC char select(char x, char y, bool c)
                ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:884:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::char2' (aka 'char2') for 1st argument
METAL_FUNC char2 select(char2 x, char2 y, bool2 c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:888:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::char3' (aka 'char3') for 1st argument
METAL_FUNC char3 select(char3 x, char3 y, bool3 c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:892:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::char4' (aka 'char4') for 1st argument
METAL_FUNC char4 select(char4 x, char4 y, bool4 c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:908:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::uchar' (aka 'unsigned char') for 1st argument
METAL_FUNC uchar select(uchar x, uchar y, bool c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:912:19: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::uchar2' (aka 'uchar2') for 1st argument
METAL_FUNC uchar2 select(uchar2 x, uchar2 y, bool2 c)
                  ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:916:19: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::uchar3' (aka 'uchar3') for 1st argument
METAL_FUNC uchar3 select(uchar3 x, uchar3 y, bool3 c)
                  ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:920:19: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::uchar4' (aka 'uchar4') for 1st argument
METAL_FUNC uchar4 select(uchar4 x, uchar4 y, bool4 c)
                  ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:936:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'short' for 1st argument
METAL_FUNC short select(short x, short y, bool c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:940:19: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::short2' (aka 'short2') for 1st argument
METAL_FUNC short2 select(short2 x, short2 y, bool2 c)
                  ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:944:19: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::short3' (aka 'short3') for 1st argument
METAL_FUNC short3 select(short3 x, short3 y, bool3 c)
                  ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:948:19: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::short4' (aka 'short4') for 1st argument
METAL_FUNC short4 select(short4 x, short4 y, bool4 c)
                  ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:964:19: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::ushort' (aka 'unsigned short') for 1st argument
METAL_FUNC ushort select(ushort x, ushort y, bool c)
                  ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:968:20: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::ushort2' (aka 'ushort2') for 1st argument
METAL_FUNC ushort2 select(ushort2 x, ushort2 y, bool2 c)
                   ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:972:20: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::ushort3' (aka 'ushort3') for 1st argument
METAL_FUNC ushort3 select(ushort3 x, ushort3 y, bool3 c)
                   ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:976:20: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::ushort4' (aka 'ushort4') for 1st argument
METAL_FUNC ushort4 select(ushort4 x, ushort4 y, bool4 c)
                   ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:992:16: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'int' for 1st argument
METAL_FUNC int select(int x, int y, bool c)
               ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:996:17: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::int2' (aka 'int2') for 1st argument
METAL_FUNC int2 select(int2 x, int2 y, bool2 c)
                ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1000:17: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::int3' (aka 'int3') for 1st argument
METAL_FUNC int3 select(int3 x, int3 y, bool3 c)
                ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1004:17: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::int4' (aka 'int4') for 1st argument
METAL_FUNC int4 select(int4 x, int4 y, bool4 c)
                ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1020:17: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::uint' (aka 'unsigned int') for 1st argument
METAL_FUNC uint select(uint x, uint y, bool c)
                ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1024:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::uint2' (aka 'uint2') for 1st argument
METAL_FUNC uint2 select(uint2 x, uint2 y, bool2 c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1028:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::uint3' (aka 'uint3') for 1st argument
METAL_FUNC uint3 select(uint3 x, uint3 y, bool3 c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1049:17: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'long' for 1st argument
METAL_FUNC long select(long x, long y, bool c)
                ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1055:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::long2' (aka 'long2') for 1st argument
METAL_FUNC long2 select(long2 x, long2 y, bool2 c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1061:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::long3' (aka 'long3') for 1st argument
METAL_FUNC long3 select(long3 x, long3 y, bool3 c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1067:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::long4' (aka 'long4') for 1st argument
METAL_FUNC long4 select(long4 x, long4 y, bool4 c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1089:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::ulong' (aka 'unsigned long') for 1st argument
METAL_FUNC ulong select(ulong x, ulong y, bool c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1095:19: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::ulong2' (aka 'ulong2') for 1st argument
METAL_FUNC ulong2 select(ulong2 x, ulong2 y, bool2 c)
                  ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1101:19: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::ulong3' (aka 'ulong3') for 1st argument
METAL_FUNC ulong3 select(ulong3 x, ulong3 y, bool3 c)
                  ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1107:19: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::ulong4' (aka 'ulong4') for 1st argument
METAL_FUNC ulong4 select(ulong4 x, ulong4 y, bool4 c)
                  ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1168:17: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'half' for 1st argument
METAL_FUNC half select(half x, half y, bool c)
                ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1172:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::half2' (aka 'half2') for 1st argument
METAL_FUNC half2 select(half2 x, half2 y, bool2 c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1176:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::half3' (aka 'half3') for 1st argument
METAL_FUNC half3 select(half3 x, half3 y, bool3 c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1180:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::half4' (aka 'half4') for 1st argument
METAL_FUNC half4 select(half4 x, half4 y, bool4 c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1196:18: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'float' for 1st argument
METAL_FUNC float select(float x, float y, bool c)
                 ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1200:19: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::float2' (aka 'float2') for 1st argument
METAL_FUNC float2 select(float2 x, float2 y, bool2 c)
                  ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1204:19: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::float3' (aka 'float3') for 1st argument
METAL_FUNC float3 select(float3 x, float3 y, bool3 c)
                  ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_relational:1208:19: note: candidate function not viable: no known conversion from 'uint4' (vector of 4 'unsigned int' values) to 'metal::float4' (aka 'float4') for 1st argument
METAL_FUNC float4 select(float4 x, float4 y, bool4 c)
                  ^
}

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

kernel void lj_clear_cells(
    device atomic_uint *cell_count [[buffer(0)]],
    constant uint      &M3         [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) 
{
    if (gid >= M3) return;
    atomic_store_explicit(&cell_count[gid], 0u, memory_order_relaxed);
}

kernel void lj_build_cells(
    device const float4 *pos          [[buffer(0)]],
    device atomic_uint  *cell_count   [[buffer(1)]],
    device       uint   *cell_list    [[buffer(2)]],
    constant uint        &N           [[buffer(3)]],
    constant uint        &M           [[buffer(4)]],
    constant float       &L           [[buffer(5)]],
    constant uint        &MAX_PER_CELL[[buffer(6)]],
    uint i [[thread_position_in_grid]]) 
{
    if (i >= N) return;
    float3 r = pos[i].xyz;
    float inv_L = 1.0f / L;
    r -= L * floor(r * inv_L);
    
    float inv_cell_size = float(M) * inv_L;
    uint cx = min(uint(r.x * inv_cell_size), M - 1u);
    uint cy = min(uint(r.y * inv_cell_size), M - 1u);
    uint cz = min(uint(r.z * inv_cell_size), M - 1u);
    
    uint cell = (cz * M + cy) * M + cx;
    uint slot = atomic_fetch_add_explicit(&cell_count[cell], 1u, memory_order_relaxed);
    if (slot < MAX_PER_CELL) {
        cell_list[cell * MAX_PER_CELL + slot] = i;
    }
}

kernel void lj_step(
    device const float4 *pos_in       [[buffer(0)]],
    device       float4 *pos_out      [[buffer(1)]],
    device const float4 *vel_in       [[buffer(2)]],
    device       float4 *vel_out      [[buffer(3)]],
    device const uint   *cell_count   [[buffer(4)]],
    device const uint   *cell_list    [[buffer(5)]],
    constant uint        &N           [[buffer(6)]],
    constant uint        &M           [[buffer(7)]],
    constant float       &L           [[buffer(8)]],
    constant float       &rcut2       [[buffer(9)]],
    constant float       &dt          [[buffer(10)]],
    constant uint        &MAX_PER_CELL[[buffer(11)]],
    uint i [[thread_position_in_grid]]) 
{
    if (i >= N) return;
    
    float3 ri = pos_in[i].xyz;
    float3 vi = vel_in[i].xyz;
    
    float inv_L = 1.0f / L;
    uint M_u = M;
    float inv_cell_size = float(M_u) * inv_L;
    
    float3 ri_w = ri - L * floor(ri * inv_L);
    uint cx = min(uint(ri_w.x * inv_cell_size), M_u - 1u);
    uint cy = min(uint(ri_w.y * inv_cell_size), M_u - 1u);
    uint cz = min(uint(ri_w.z * inv_cell_size), M_u - 1u);

    float cell_size = L / float(M_u);
    float rx = ri_w.x - float(cx) * cell_size;
    float ry = ri_w.y - float(cy) * cell_size;
    float rz = ri_w.z - float(cz) * cell_size;

    float dx_sq[3] = { rx * rx, 0.0f, (cell_size - rx) * (cell_size - rx) };
    float dy_sq[3] = { ry * ry, 0.0f, (cell_size - ry) * (cell_size - ry) };
    float dz_sq[3] = { rz * rz, 0.0f, (cell_size - rz) * (cell_size - rz) };

    uint cx_n[3] = { cx == 0 ? M_u - 1 : cx - 1, cx, cx == M_u - 1 ? 0 : cx + 1 };
    uint cy_n[3] = { cy == 0 ? M_u - 1 : cy - 1, cy, cy == M_u - 1 ? 0 : cy + 1 };
    uint cz_n[3] = { cz == 0 ? M_u - 1 : cz - 1, cz, cz == M_u - 1 ? 0 : cz + 1 };

    float3 a = float3(0.0f);

    #pragma unroll
    for (int z = 0; z < 3; ++z) {
        float z2 = dz_sq[z];
        uint cz_idx = cz_n[z] * M_u * M_u;
        
        #pragma unroll
        for (int y = 0; y < 3; ++y) {
            float zy2 = z2 + dy_sq[y];
            if (zy2 >= rcut2) continue;
            uint cy_idx = cz_idx + cy_n[y] * M_u;
            
            #pragma unroll
            for (int x = 0; x < 3; ++x) {
                float zyx2 = zy2 + dx_sq[x];
                if (zyx2 >= rcut2) continue;
                
                uint nc = cy_idx + cx_n[x];
                uint cnt = min(cell_count[nc], MAX_PER_CELL);
                if (cnt == 0) continue;
                
                uint cell_base = nc * MAX_PER_CELL;
                device const uint4* cell_list_u4 = (device const uint4*)(cell_list + cell_base);
                uint num_chunks = (cnt + 3) / 4;
                
                for (uint k = 0; k < num_chunks; ++k) {
                    uint4 j4 = cell_list_u4[k];
                    uint base_p = k * 4;
                    
                    uint j_safe[4];
                    j_safe[0] = (base_p + 0 < cnt) ? j4.x : i;
                    j_safe[1] = (base_p + 1 < cnt) ? j4.y : i;
                    j_safe[2] = (base_p + 2 < cnt) ? j4.z : i;
                    j_safe[3] = (base_p + 3 < cnt) ? j4.w : i;
                    
                    float3 p_val[4];
                    #pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        p_val[p] = pos_in[j_safe[p]].xyz;
                    }
                    
                    float3 dx[4];
                    float r2[4];
                    #pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        dx[p] = p_val[p] - ri;
                        dx[p] -= L * rint(dx[p] * inv_L);
                        r2[p] = dot(dx[p], dx[p]);
                    }
                    
                    #pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        if (r2[p] < rcut2 && r2[p] > 1e-6f) {
                            float inv_r2 = 1.0f / r2[p];
                            float inv_r6 = inv_r2 * inv_r2 * inv_r2;
                            a += (-24.0f * inv_r6 * (2.0f * inv_r6 - 1.0f) * inv_r2) * dx[p];
                        }
                    }
                }
            }
        }
    }

    float3 v_new = vi + a * dt;
    float3 r_new = ri + v_new * dt;
    pos_out[i] = float4(r_new, 0.0f);
    vel_out[i] = float4(v_new, 0.0f);
}
```

Incumbent result:
  N1728_M5_steps20: correct, 2.28 ms, 10.2 GFLOPS (useful pairs only) (0.2% of 4500 GFLOPS)
  N4096_M7_steps15: correct, 1.32 ms, 31.3 GFLOPS (useful pairs only) (0.7% of 4500 GFLOPS)
  N10648_M10_steps10: correct, 2.21 ms, 32.4 GFLOPS (useful pairs only) (0.7% of 4500 GFLOPS)
  score (gmean of fraction): 0.0048

## History

- iter  0: compile=OK | correct=True | score=0.0028461040676860555
- iter  1: compile=OK | correct=True | score=0.004286953578398174
- iter  2: compile=OK | correct=True | score=0.004296831464338089
- iter  3: compile=OK | correct=True | score=0.0029830433221850893
- iter  4: compile=OK | correct=True | score=0.004839503522282914
- iter  5: compile=OK | correct=True | score=0.0037016542173682575
- iter  6: compile=FAIL | correct=False | score=N/A
- iter  7: compile=FAIL | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
