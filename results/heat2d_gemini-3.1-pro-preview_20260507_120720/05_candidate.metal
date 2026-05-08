#include <metal_stdlib>
using namespace metal;

kernel void heat_step(device const float *u_in  [[buffer(0)]],
                      device       float *u_out [[buffer(1)]],
                      constant uint      &NX    [[buffer(2)]],
                      constant uint      &NY    [[buffer(3)]],
                      constant float     &alpha [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]],
                      uint2 tid [[thread_position_in_threadgroup]],
                      uint2 tsize [[threads_per_threadgroup]]) {
    
    // Allocate shared memory for up to 1024 threads.
    // Max tile size occurs at a 1024x1 threadgroup: (1024+2)*(1+2) = 3078 elements.
    threadgroup float tile[3078];
    
    uint nx = NX;
    uint ny = NY;
    uint i = gid.x;
    uint j = gid.y;
    
    bool valid = (i < nx && j < ny);
    
    uint stride = tsize.x + 2;
    #define TILE_IDX(y, x) ((y) * stride + (x))
    
    // 1. Center load (also kept in register for reuse)
    float c_val = valid ? u_in[j * nx + i] : 0.0f;
    tile[TILE_IDX(tid.y + 1, tid.x + 1)] = c_val;
    
    // 2. Halo loads (clamps strictly bounds-checked to avoid page faults)
    if (tid.y == 0) {
        uint y = (j > 0) ? j - 1 : 0;
        tile[TILE_IDX(0, tid.x + 1)] = valid ? u_in[y * nx + i] : 0.0f;
    }
    if (tid.y == tsize.y - 1) {
        uint y = (j < ny - 1) ? j + 1 : j;
        tile[TILE_IDX(tsize.y + 1, tid.x + 1)] = valid ? u_in[y * nx + i] : 0.0f;
    }
    
    if (tid.x == 0) {
        uint x = (i > 0) ? i - 1 : 0;
        tile[TILE_IDX(tid.y + 1, 0)] = valid ? u_in[j * nx + x] : 0.0f;
    }
    if (tid.x == tsize.x - 1) {
        uint x = (i < nx - 1) ? i + 1 : i;
        tile[TILE_IDX(tid.y + 1, tsize.x + 1)] = valid ? u_in[j * nx + x] : 0.0f;
    }
    
    // 3. Synchronize threadgroup memory visibility
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // 4. Out-of-bounds threads safely exit after participating in the barrier
    if (!valid) return;
    
    uint idx = j * nx + i;
    
    // 5. Dirichlet boundaries bypass the stencil update
    if (i == 0 || j == 0 || i == nx - 1 || j == ny - 1) {
        u_out[idx] = c_val;
        return;
    }
    
    // 6. Stencil computation for interior cells
    float l = tile[TILE_IDX(tid.y + 1, tid.x)];
    float r = tile[TILE_IDX(tid.y + 1, tid.x + 2)];
    float t = tile[TILE_IDX(tid.y,     tid.x + 1)];
    float b = tile[TILE_IDX(tid.y + 2, tid.x + 1)];
    
    float a = alpha;
    u_out[idx] = c_val + a * (l + r + t + b - 4.0f * c_val);
    
    #undef TILE_IDX
}