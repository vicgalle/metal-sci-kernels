// Naive seed for a Morton-ordered 3D heat-equation stencil.
//
// State is a 1D float buffer of length N^3 (N a power of 2), indexed in
// Z-order (Morton) layout: the value at coordinates (x, y, z) ∈ [0, N)^3
// lives at linear index
//
//     M(x, y, z) = sum_{i=0}^{logN-1} ( x[i]·2^(3i)
//                                       + y[i]·2^(3i+1)
//                                       + z[i]·2^(3i+2) )
//
// where x[i] is the i-th bit of x and logN = log2(N). M is a bijection
// onto [0, N^3) so the buffer is exactly N^3 floats with no padding.
//
// One timestep, forward Euler, 7-point Laplacian:
//
//   u_new[M(x,y,z)] = u[M(x,y,z)] + alpha * ( u[M(x-1,y,z)] + u[M(x+1,y,z)]
//                                           + u[M(x,y-1,z)] + u[M(x,y+1,z)]
//                                           + u[M(x,y,z-1)] + u[M(x,y,z+1)]
//                                           - 6 u[M(x,y,z)] )
//
// Dirichlet boundary: a cell with x, y, or z in {0, N-1} (i.e. on a face
// of the cube) copies u → u_new unchanged. The host ping-pongs u_in and
// u_out across n_steps timesteps in one command buffer.
//
// Buffer layout:
//   buffer 0: const float* u_in     (N^3, Morton-ordered)
//   buffer 1: device float* u_out   (N^3, Morton-ordered)
//   buffer 2: const uint& N         (grid size, power of 2)
//   buffer 3: const uint& logN      (= log2(N); host guarantees N == 1<<logN)
//   buffer 4: const float& alpha    (alpha = dt / dx^2 ∈ [0, 1/6])
//
// Convention: tid is the MORTON INDEX. Decode tid → (x, y, z) for the
// boundary check, then compute the Morton indices of the six neighbours
// to gather the stencil. The candidate is expected to find a faster
// Morton encode/decode (magic-constant bit-spread or lookup table) and/or
// neighbour-index bit twiddling that avoids the full encode round-trip.

#include <metal_stdlib>
using namespace metal;

inline uint morton_encode_3d(uint x, uint y, uint z, uint logN) {
    uint m = 0u;
    for (uint i = 0u; i < logN; ++i) {
        m |= ((x >> i) & 1u) << (3u * i + 0u);
        m |= ((y >> i) & 1u) << (3u * i + 1u);
        m |= ((z >> i) & 1u) << (3u * i + 2u);
    }
    return m;
}

inline void morton_decode_3d(uint m, uint logN,
                             thread uint &x, thread uint &y, thread uint &z) {
    x = 0u; y = 0u; z = 0u;
    for (uint i = 0u; i < logN; ++i) {
        x |= ((m >> (3u * i + 0u)) & 1u) << i;
        y |= ((m >> (3u * i + 1u)) & 1u) << i;
        z |= ((m >> (3u * i + 2u)) & 1u) << i;
    }
}

kernel void morton_stencil(device const float *u_in   [[buffer(0)]],
                           device       float *u_out  [[buffer(1)]],
                           constant uint      &N      [[buffer(2)]],
                           constant uint      &logN   [[buffer(3)]],
                           constant float     &alpha  [[buffer(4)]],
                           uint tid [[thread_position_in_grid]])
{
    uint total = N * N * N;
    if (tid >= total) return;

    uint x, y, z;
    morton_decode_3d(tid, logN, x, y, z);

    // Dirichlet BC: faces of the cube copy through unchanged.
    if (x == 0u || y == 0u || z == 0u ||
        x == N - 1u || y == N - 1u || z == N - 1u) {
        u_out[tid] = u_in[tid];
        return;
    }

    uint m_xm = morton_encode_3d(x - 1u, y,      z,      logN);
    uint m_xp = morton_encode_3d(x + 1u, y,      z,      logN);
    uint m_ym = morton_encode_3d(x,      y - 1u, z,      logN);
    uint m_yp = morton_encode_3d(x,      y + 1u, z,      logN);
    uint m_zm = morton_encode_3d(x,      y,      z - 1u, logN);
    uint m_zp = morton_encode_3d(x,      y,      z + 1u, logN);

    float c  = u_in[tid];
    float xm = u_in[m_xm];
    float xp = u_in[m_xp];
    float ym = u_in[m_ym];
    float yp = u_in[m_yp];
    float zm = u_in[m_zm];
    float zp = u_in[m_zp];

    u_out[tid] = c + alpha * (xm + xp + ym + yp + zm + zp - 6.0f * c);
}
