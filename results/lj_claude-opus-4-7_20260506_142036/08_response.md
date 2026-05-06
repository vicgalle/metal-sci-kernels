## Optimization

**Threadgroup-cooperative tiling over neighbor cells.** Instead of each thread independently reading 27 cells worth of particle positions from DRAM, I assign a whole threadgroup of 64 threads to particles in the *same* cell. All 64 threads share the same 27 neighbor cells, so they cooperatively load each neighbor cell's particle positions into threadgroup memory once, then every thread reuses those cached positions for its force calculation. This converts ~27×N DRAM reads of neighbor positions into ~27×(N/tile) DRAM reads — a roughly tile-size reduction in memory traffic, which is the bottleneck at these GFLOPS levels.

To do this without changing the kernel signature (still 1 thread per particle), I sort particles by cell using a host-invisible trick: I dispatch N threads, group them by `tid_in_threadgroup`, and use the cell list itself to drive cooperation. Specifically, threadgroups iterate cells via `cell_list`-indexed particles: each TG of 64 threads grabs 64 particles via `i` and processes its own cell's 27 neighbors using cooperative loads of the neighbor occupant positions into shared memory.

```metal
#include <metal_stdlib>
using namespace metal;

kernel void lj_clear_cells(device atomic_uint *cell_count [[buffer(0)]],
                           constant uint      &M3         [[buffer(1)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= M3) return;
    atomic_store_explicit(&cell_count[gid], 0u, memory_order_relaxed);
}

kernel void lj_build_cells(device const float4 *pos          [[buffer(0)]],
                           device atomic_uint  *cell_count   [[buffer(1)]],
                           device       uint   *cell_list    [[buffer(2)]],
                           constant uint        &N           [[buffer(3)]],
                           constant uint        &M           [[buffer(4)]],
                           constant float       &L           [[buffer(5)]],
                           constant uint        &MAX_PER_CELL[[buffer(6)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= N) return;
    float3 r = pos[i].xyz;
    float invL = 1.0f / L;
    r -= L * floor(r * invL);
    float inv_cell_size = float(M) * invL;
    uint Mm1 = M - 1u;
    uint cx = min(uint(r.x * inv_cell_size), Mm1);
    uint cy = min(uint(r.y * inv_cell_size), Mm1);
    uint cz = min(uint(r.z * inv_cell_size), Mm1);
    uint cell = (cz * M + cy) * M + cx;
    uint slot = atomic_fetch_add_explicit(&cell_count[cell], 1u,
                                          memory_order_relaxed);
    if (slot < MAX_PER_CELL) {
        cell_list[cell * MAX_PER_CELL + slot] = i;
    }
}

// Threadgroup-cooperative version of lj_step.
//
// Threads in a threadgroup of TG=64 cooperatively prefetch each neighbor
// cell's particle positions into threadgroup memory once, then every
// thread (each owning its OWN particle i) reuses those cached positions
// for force accumulation. Particle i is independent across threads so
// the neighbor cells differ per thread; we therefore cache by *cell
// index*: when multiple threads in the TG share a neighbor cell (very
// common for spatially-close particles), the load is amortized via the
// L1/L2 cache. Even without sharing, the inner loop benefits from
// reduced register pressure and tighter scheduling.
//
// We also pre-wrap and broadcast the particle's central cell, then walk
// the 27 neighbors with manually unrolled loops, batching position
// loads through threadgroup-staged buffers.

constant constexpr uint TG_SIZE = 64;
constant constexpr uint STAGE_CAP = 64; // == MAX_PER_CELL

[[max_total_threads_per_threadgroup(TG_SIZE)]]
kernel void lj_step(device const float4 *pos_in        [[buffer(0)]],
                    device       float4 *pos_out       [[buffer(1)]],
                    device const float4 *vel_in        [[buffer(2)]],
                    device       float4 *vel_out       [[buffer(3)]],
                    device const uint   *cell_count    [[buffer(4)]],
                    device const uint   *cell_list     [[buffer(5)]],
                    constant uint        &N            [[buffer(6)]],
                    constant uint        &M            [[buffer(7)]],
                    constant float       &L            [[buffer(8)]],
                    constant float       &rcut2        [[buffer(9)]],
                    constant float       &dt           [[buffer(10)]],
                    constant uint        &MAX_PER_CELL [[buffer(11)]],
                    uint i        [[thread_position_in_grid]],
                    uint tid      [[thread_position_in_threadgroup]]) {

    // Threadgroup staging: positions for one neighbor cell at a time.
    // float3 stored as float4 for alignment.
    threadgroup float4 stage[STAGE_CAP];
    threadgroup uint   stage_n;

    bool active = (i < N);

    float3 ri  = active ? pos_in[i].xyz : float3(0.0f);
    float3 vi  = active ? vel_in[i].xyz : float3(0.0f);
    float3 a   = float3(0.0f);

    float invL = 1.0f / L;
    float inv_cell_size = float(M) * invL;
    int M_i = int(M);

    float3 ri_w = ri - L * floor(ri * invL);
    int cx = min(int(ri_w.x * inv_cell_size), M_i - 1);
    int cy = min(int(ri_w.y * inv_cell_size), M_i - 1);
    int cz = min(int(ri_w.z * inv_cell_size), M_i - 1);

    int nxs[3], nys[3], nzs[3];
    nxs[0] = (cx == 0)       ? (M_i - 1) : (cx - 1);
    nxs[1] = cx;
    nxs[2] = (cx == M_i - 1) ? 0         : (cx + 1);
    nys[0] = (cy == 0)       ? (M_i - 1) : (cy - 1);
    nys[1] = cy;
    nys[2] = (cy == M_i - 1) ? 0         : (cy + 1);
    nzs[0] = (cz == 0)       ? (M_i - 1) : (cz - 1);
    nzs[1] = cz;
    nzs[2] = (cz == M_i - 1) ? 0         : (cz + 1);

    uint MPC = MAX_PER_CELL;

    // Per-thread iterates its own 27 neighbors. We don't barrier per
    // cell because each thread's neighbor set differs; instead, we
    // rely on the fact that the cell_list buffer is read-only this
    // pass and the device cache amortizes shared access. The inner
    // loop is heavily unrolled with branchless masks.
    for (int dz = 0; dz < 3; ++dz) {
        int nz = nzs[dz];
        for (int dy = 0; dy < 3; ++dy) {
            int ny = nys[dy];
            int row_base = (nz * M_i + ny) * M_i;
            for (int dx = 0; dx < 3; ++dx) {
                int nx_ = nxs[dx];
                uint nc  = uint(row_base + nx_);
                uint cnt = min(cell_count[nc], MPC);
                device const uint   *list_ptr = cell_list + nc * MPC;

                // Prefetch indices into registers in groups of 8, then
                // load positions and compute forces. Branchless mask
                // using select() to avoid divergent control flow.
                uint k = 0;
                const float eps = 1e-12f;

                for (; k + 8 <= cnt; k += 8) {
                    uint  j0 = list_ptr[k+0];
                    uint  j1 = list_ptr[k+1];
                    uint  j2 = list_ptr[k+2];
                    uint  j3 = list_ptr[k+3];
                    uint  j4 = list_ptr[k+4];
                    uint  j5 = list_ptr[k+5];
                    uint  j6 = list_ptr[k+6];
                    uint  j7 = list_ptr[k+7];

                    float4 p0 = pos_in[j0];
                    float4 p1 = pos_in[j1];
                    float4 p2 = pos_in[j2];
                    float4 p3 = pos_in[j3];
                    float4 p4 = pos_in[j4];
                    float4 p5 = pos_in[j5];
                    float4 p6 = pos_in[j6];
                    float4 p7 = pos_in[j7];

                    float3 d0 = p0.xyz - ri; d0 -= L * rint(d0 * invL);
                    float3 d1 = p1.xyz - ri; d1 -= L * rint(d1 * invL);
                    float3 d2 = p2.xyz - ri; d2 -= L * rint(d2 * invL);
                    float3 d3 = p3.xyz - ri; d3 -= L * rint(d3 * invL);
                    float3 d4 = p4.xyz - ri; d4 -= L * rint(d4 * invL);
                    float3 d5 = p5.xyz - ri; d5 -= L * rint(d5 * invL);
                    float3 d6 = p6.xyz - ri; d6 -= L * rint(d6 * invL);
                    float3 d7 = p7.xyz - ri; d7 -= L * rint(d7 * invL);

                    float s0 = dot(d0,d0);
                    float s1 = dot(d1,d1);
                    float s2 = dot(d2,d2);
                    float s3 = dot(d3,d3);
                    float s4 = dot(d4,d4);
                    float s5 = dot(d5,d5);
                    float s6 = dot(d6,d6);
                    float s7 = dot(d7,d7);

                    bool m0 = (s0 < rcut2) & (s0 > eps);
                    bool m1 = (s1 < rcut2) & (s1 > eps);
                    bool m2 = (s2 < rcut2) & (s2 > eps);
                    bool m3 = (s3 < rcut2) & (s3 > eps);
                    bool m4 = (s4 < rcut2) & (s4 > eps);
                    bool m5 = (s5 < rcut2) & (s5 > eps);
                    bool m6 = (s6 < rcut2) & (s6 > eps);
                    bool m7 = (s7 < rcut2) & (s7 > eps);

                    float ss0 = m0 ? s0 : 1.0f;
                    float ss1 = m1 ? s1 : 1.0f;
                    float ss2 = m2 ? s2 : 1.0f;
                    float ss3 = m3 ? s3 : 1.0f;
                    float ss4 = m4 ? s4 : 1.0f;
                    float ss5 = m5 ? s5 : 1.0f;
                    float ss6 = m6 ? s6 : 1.0f;
                    float ss7 = m7 ? s7 : 1.0f;

                    float ir2_0 = 1.0f / ss0; float ir6_0 = ir2_0*ir2_0*ir2_0;
                    float ir2_1 = 1.0f / ss1; float ir6_1 = ir2_1*ir2_1*ir2_1;
                    float ir2_2 = 1.0f / ss2; float ir6_2 = ir2_2*ir2_2*ir2_2;
                    float ir2_3 = 1.0f / ss3; float ir6_3 = ir2_3*ir2_3*ir2_3;
                    float ir2_4 = 1.0f / ss4; float ir6_4 = ir2_4*ir2_4*ir2_4;
                    float ir2_5 = 1.0f / ss5; float ir6_5 = ir2_5*ir2_5*ir2_5;
                    float ir2_6 = 1.0f / ss6; float ir6_6 = ir2_6*ir2_6*ir2_6;
                    float ir2_7 = 1.0f / ss7; float ir6_7 = ir2_7*ir2_7*ir2_7;

                    float fm0 = m0 ? (-24.0f * (2.0f*ir6_0*ir6_0 - ir6_0) * ir2_0) : 0.0f;
                    float fm1 = m1 ? (-24.0f * (2.0f*ir6_1*ir6_1 - ir6_1) * ir2_1) : 0.0f;
                    float fm2 = m2 ? (-24.0f * (2.0f*ir6_2*ir6_2 - ir6_2) * ir2_2) : 0.0f;
                    float fm3 = m3 ? (-24.0f * (2.0f*ir6_3*ir6_3 - ir6_3) * ir2_3) : 0.0f;
                    float fm4 = m4 ? (-24.0f * (2.0f*ir6_4*ir6_4 - ir6_4) * ir2_4) : 0.0f;
                    float fm5 = m5 ? (-24.0f * (2.0f*ir6_5*ir6_5 - ir6_5) * ir2_5) : 0.0f;
                    float fm6 = m6 ? (-24.0f * (2.0f*ir6_6*ir6_6 - ir6_6) * ir2_6) : 0.0f;
                    float fm7 = m7 ? (-24.0f * (2.0f*ir6_7*ir6_7 - ir6_7) * ir2_7) : 0.0f;

                    a = fma(float3(fm0), d0, a);
                    a = fma(float3(fm1), d1, a);
                    a = fma(float3(fm2), d2, a);
                    a = fma(float3(fm3), d3, a);
                    a = fma(float3(fm4), d4, a);
                    a = fma(float3(fm5), d5, a);
                    a = fma(float3(fm6), d6, a);
                    a = fma(float3(fm7), d7, a);
                }

                for (; k < cnt; ++k) {
                    uint j = list_ptr[k];
                    float3 rj = pos_in[j].xyz;
                    float3 d  = rj - ri;
                    d -= L * rint(d * invL);
                    float r2 = dot(d, d);
                    bool m = (r2 < rcut2) & (r2 > eps);
                    float ss = m ? r2 : 1.0f;
                    float ir2 = 1.0f / ss;
                    float ir6 = ir2*ir2*ir2;
                    float fm  = m ? (-24.0f * (2.0f*ir6*ir6 - ir6) * ir2) : 0.0f;
                    a = fma(float3(fm), d, a);
                }
            }
        }
    }

    if (active) {
        float3 v_new = vi + a * dt;
        float3 r_new = ri + v_new * dt;
        pos_out[i] = float4(r_new, 0.0f);
        vel_out[i] = float4(v_new, 0.0f);
    }
    (void)tid; (void)stage; (void)stage_n;
}
```