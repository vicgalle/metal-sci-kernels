To optimize this memory-bound 5-point stencil kernel, we apply **threadgroup-memory tiling**. 

1. **Optimization applied:** We use a `threadgroup` array to cache the center cell `c` fetched by each thread. Neighboring cells (`l, r, d, u`) are then read directly from this fast shared memory if they fall within the same threadgroup tile. If a neighbor lies across the threadgroup boundary, the thread falls back to loading it from global memory (L1 cache). We also cache uniform constants (`NX, NY, alpha`) in registers.
2. **Expected improvement:** The baseline performs 5 redundant global memory loads (L1 cache requests) per thread. This threadgroup tiling approach cooperatively shares the fetched values, reducing the global memory loads to ~1.31 per thread. This substantially lowers L1 cache bandwidth pressure and instruction count, leading to higher effective throughput. Barrier safety is guaranteed by keeping inactive threads alive with a safe `idx=0` read until the barrier completes.

```metal
#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(1024)]]
kernel void heat_step(device const float *u_in  [[buffer(0)]],
                      device       float *u_out [[buffer(1)]],
                      constant uint      &NX    [[buffer(2)]],
                      constant uint      &NY    [[buffer(3)]],
                      constant float     &alpha [[buffer(4)]],
                      uint2 gid   [[thread_position_in_grid]],
                      uint2 tid   [[thread_position_in_threadgroup]],
                      uint2 tsize [[threads_per_threadgroup]]) {
    uint nx = NX;
    uint ny = NY;
    uint i = gid.x;
    uint j = gid.y;
    
    // Determine validity but do NOT early return yet to avoid barrier deadlocks
    bool valid = (i < nx && j < ny);
    
    // Safely prevent out-of-bounds reads for inactive threads by clamping to index 0
    uint idx = valid ? (j * nx + i) : 0;
    float c = valid ? u_in[idx] : 0.0f;

    // Use a 1D threadgroup array (max 1024 threads per group on Apple Silicon)
    threadgroup float tile[1024];
    uint tg_idx = tid.y * tsize.x + tid.x;
    tile[tg_idx] = c;

    // Synchronize to ensure all threads have populated the tile
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Now it is safe for out-of-bounds threads to exit
    if (!valid) return;

    // Dirichlet boundary condition
    if (i == 0 || j == 0 || i == nx - 1 || j == ny - 1) {
        u_out[idx] = c;
    } else {
        // Fetch neighbors from the threadgroup tile; fallback to global memory at tile boundaries
        float l = (tid.x > 0)             ? tile[tg_idx - 1]       : u_in[idx - 1];
        float r = (tid.x < tsize.x - 1)   ? tile[tg_idx + 1]       : u_in[idx + 1];
        float u = (tid.y > 0)             ? tile[tg_idx - tsize.x] : u_in[idx - nx];
        float d = (tid.y < tsize.y - 1)   ? tile[tg_idx + tsize.x] : u_in[idx + nx];
        
        float a = alpha;
        u_out[idx] = c + a * (l + r + d + u - 4.0f * c);
    }
}
```