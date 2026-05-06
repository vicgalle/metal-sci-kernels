"""Lennard-Jones MD with cell-list spatial hash (Plan §4, Task 6).

Three-kernel suite, dispatched in sequence each timestep:
  1) ``lj_clear_cells`` — zero the per-cell occupancy counter.
  2) ``lj_build_cells`` — atomic-scatter particle indices into their cell.
  3) ``lj_step``        — compute LJ forces from the 27 neighbour cells
     (minimum-image PBC, hard cutoff at rcut), then take one symplectic-Euler
     step (v += a dt; r += v dt).

Storage is float4 SoA-padded for positions/velocities (matching the nbody
task), with cell_count[M^3] (uint) and cell_list[M^3 * MAX_PER_CELL] (uint).
The host ping-pongs (pos_in, pos_out) and (vel_in, vel_out) across n_steps,
encoding all dispatches into one command buffer for accurate end-to-end
GPU timing.

Correctness is checked against a chunked CPU reference: forces summed in
fp64 (eliminating CPU summation noise), then downcast to fp32 for the
integration step so we match the GPU's storage precision.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import Metal

from ..harness import MetalHarness
from ..hardware import ChipSpec
from ..task import (
    Task, TaskSize, TaskSpec, SizeResult,
    gflops, register_task,
)


_SEED = Path(__file__).resolve().parent.parent.parent / "seeds" / "lj.metal"

# Generous bound on per-cell occupancy at the densities used here
# (rho ~ 0.66 at spacing 1.15, expected ~ rho * cell_size^3 ~ 12-22).
# Particles overflowing this bound are silently dropped by build_cells; the
# initial state is constructed so this does not happen.
_MAX_PER_CELL = 64


def _make_init(N: int, L: float, seed: int = 0xCAFE):
    """Cubic K^3 = N grid + small perturbation + small random velocities.

    Returns (pos3, vel3, pos4, vel4) — fp32. pos4/vel4 are .xyz-padded for
    Metal's float4 layout. The position perturbation breaks the lattice's
    force symmetry; the small initial velocities ensure that even a
    completely no-op step kernel produces a position difference O(v*dt*n)
    that is comfortably above any plausible fp32 ordering noise.
    """
    K = round(N ** (1.0 / 3.0))
    if K * K * K != N:
        raise ValueError(f"N={N} must be a perfect cube")
    spacing = L / K
    rng = np.random.default_rng(seed)
    grid = np.indices((K, K, K), dtype=np.float32) * np.float32(spacing)
    pos = grid.transpose(1, 2, 3, 0).reshape(-1, 3)
    perturb = rng.uniform(
        -0.05 * spacing, 0.05 * spacing, size=pos.shape,
    ).astype(np.float32)
    pos = pos + perturb
    pos = pos - np.float32(L) * np.floor(pos / np.float32(L))  # wrap to [0, L)
    pos = np.ascontiguousarray(pos.astype(np.float32))
    # Net-zero linear momentum; |v| ~ 0.3 per component gives displacement
    # ~v*dt*n_steps ~ 1.5e-2..3e-2 over the n_steps used here, ~100x the
    # fp32 ordering noise tolerance below.
    vel = rng.uniform(-0.3, 0.3, size=pos.shape).astype(np.float32)
    vel -= vel.mean(axis=0, keepdims=True).astype(np.float32)
    pos4 = np.zeros((N, 4), dtype=np.float32)
    pos4[:, :3] = pos
    vel4 = np.zeros((N, 4), dtype=np.float32)
    vel4[:, :3] = vel
    return pos, vel, pos4, vel4


def _compute_forces_chunked(pos_fp32: np.ndarray, L: float, rcut2: float,
                            chunk: int = 256) -> np.ndarray:
    """LJ accelerations from chunked brute-force pair scan, fp64 internally.

    Both CPU and GPU iterate the SAME set of pairs (those with min-image
    distance < rcut), so the only source of disagreement is fp32 summation
    order. We sum in fp64 here so the reference is essentially exact;
    tolerances downstream then absorb the GPU's fp32 noise.
    """
    N = pos_fp32.shape[0]
    pos = pos_fp32.astype(np.float64)
    a = np.zeros((N, 3), dtype=np.float64)
    L64 = float(L)
    for i0 in range(0, N, chunk):
        i1 = min(i0 + chunk, N)
        d = pos[None, :, :] - pos[i0:i1, None, :]                # (B, N, 3)
        d -= L64 * np.round(d / L64)                              # min-image
        r2 = (d * d).sum(axis=-1)                                 # (B, N)
        mask = (r2 < rcut2) & (r2 > 1e-12)
        r2_safe = np.where(mask, r2, 1.0)
        inv_r2 = 1.0 / r2_safe
        inv_r6 = inv_r2 ** 3
        inv_r12 = inv_r6 ** 2
        # F_on_i = -24 (2/r^12 - 1/r^6) / r^2 * (r_j - r_i)
        fmag = -24.0 * (2.0 * inv_r12 - inv_r6) * inv_r2
        fmag = np.where(mask, fmag, 0.0)
        a[i0:i1] = (fmag[:, :, None] * d).sum(axis=1)
    return a.astype(np.float32)


def _cpu_reference(pos0: np.ndarray, vel0: np.ndarray,
                   L: float, rcut: float, dt: float, n_steps: int):
    """Symplectic-Euler reference trajectory. Forces in fp64, state in fp32."""
    pos = pos0.astype(np.float32, copy=True)
    vel = vel0.astype(np.float32, copy=True)
    rcut2 = float(rcut * rcut)
    dt32 = np.float32(dt)
    for _ in range(n_steps):
        a = _compute_forces_chunked(pos, L, rcut2)
        vel = (vel + a * dt32).astype(np.float32)
        pos = (pos + vel * dt32).astype(np.float32)
    return pos, vel


def _count_pairs_within_rcut(pos: np.ndarray, L: float, rcut: float,
                             chunk: int = 256) -> int:
    """Number of unordered pairs {i, j}, i != j, with min-image |r_i - r_j| < rcut."""
    N = pos.shape[0]
    rcut2 = float(rcut * rcut)
    pos64 = pos.astype(np.float64)
    L64 = float(L)
    total_ordered = 0
    for i0 in range(0, N, chunk):
        i1 = min(i0 + chunk, N)
        d = pos64[None, :, :] - pos64[i0:i1, None, :]
        d -= L64 * np.round(d / L64)
        r2 = (d * d).sum(axis=-1)
        mask = (r2 < rcut2) & (r2 > 1e-12)
        total_ordered += int(mask.sum())
    return total_ordered // 2


@register_task("lj")
class LJMDTask(Task):
    spec = TaskSpec(
        name="lj",
        description=(
            "Lennard-Jones molecular dynamics with a cell-list spatial hash. "
            "Cubic periodic box of side L; cutoff rcut = 2.5 (sigma = epsilon "
            "= mass = 1).\n\n"
            "Per timestep, three kernels are dispatched in this fixed order:\n"
            "  1) lj_clear_cells: zero the per-cell occupancy counter (M^3 "
            "threads).\n"
            "  2) lj_build_cells: each particle thread computes its cell "
            "index (after wrapping its position into [0, L)) and atomically "
            "appends itself to that cell (N threads).\n"
            "  3) lj_step: each particle thread iterates the 27 neighbour "
            "cells (its own cell + 3^3 - 1 face/edge/corner neighbours, with "
            "periodic wrap), reads each occupant from cell_list, and sums the "
            "Lennard-Jones force from those within rcut. It then takes one "
            "symplectic-Euler step:  v_new = v + a*dt; r_new = r + v_new*dt "
            "(N threads).\n\n"
            "Cell layout: M cells per side; cell index = (cz*M + cy)*M + cx; "
            "cell_size = L/M is guaranteed >= rcut so 27 neighbour cells "
            "cover all interactions. cell_count[M^3] holds the per-cell "
            "occupancy, cell_list[M^3 * MAX_PER_CELL] holds the particle "
            "indices, with row-major slot order. MAX_PER_CELL = 64 is "
            "generous for the supplied initial states; particles exceeding "
            "this cap are silently dropped (the seed tolerates this since "
            "the well-conditioned initial state never overflows, and a "
            "candidate may rely on the same invariant).\n\n"
            "Lennard-Jones force on i from j (sigma = epsilon = 1):\n"
            "  d = (r_j - r_i), minimum-image:  d -= L * round(d / L)\n"
            "  r2 = dot(d, d); skip if r2 >= rcut^2 or r2 ~= 0\n"
            "  inv_r2 = 1/r2; inv_r6 = inv_r2^3; inv_r12 = inv_r6^2\n"
            "  F_on_i = -24 * (2*inv_r12 - inv_r6) * inv_r2 * d\n"
            "  a_i = sum of F_on_i over all j within cutoff (mass = 1).\n\n"
            "Positions/velocities are stored as float4 with .xyz holding the "
            "data and .w padding (matches the nbody task's layout). The host "
            "ping-pongs (pos_in, pos_out) and (vel_in, vel_out) buffer pairs "
            "each step; cell_count and cell_list are scratch buffers reused "
            "every step (cleared by lj_clear_cells)."
        ),
        kernel_signatures=(
            "kernel void lj_clear_cells(\n"
            "    device atomic_uint *cell_count [[buffer(0)]],\n"
            "    constant uint      &M3         [[buffer(1)]],\n"
            "    uint gid [[thread_position_in_grid]]);\n"
            "\n"
            "kernel void lj_build_cells(\n"
            "    device const float4 *pos          [[buffer(0)]],\n"
            "    device atomic_uint  *cell_count   [[buffer(1)]],\n"
            "    device       uint   *cell_list    [[buffer(2)]],\n"
            "    constant uint        &N           [[buffer(3)]],\n"
            "    constant uint        &M           [[buffer(4)]],\n"
            "    constant float       &L           [[buffer(5)]],\n"
            "    constant uint        &MAX_PER_CELL[[buffer(6)]],\n"
            "    uint i [[thread_position_in_grid]]);\n"
            "\n"
            "kernel void lj_step(\n"
            "    device const float4 *pos_in       [[buffer(0)]],\n"
            "    device       float4 *pos_out      [[buffer(1)]],\n"
            "    device const float4 *vel_in       [[buffer(2)]],\n"
            "    device       float4 *vel_out      [[buffer(3)]],\n"
            "    device const uint   *cell_count   [[buffer(4)]],\n"
            "    device const uint   *cell_list    [[buffer(5)]],\n"
            "    constant uint        &N           [[buffer(6)]],\n"
            "    constant uint        &M           [[buffer(7)]],\n"
            "    constant float       &L           [[buffer(8)]],\n"
            "    constant float       &rcut2       [[buffer(9)]],\n"
            "    constant float       &dt          [[buffer(10)]],\n"
            "    constant uint        &MAX_PER_CELL[[buffer(11)]],\n"
            "    uint i [[thread_position_in_grid]]);\n"
            "\n"
            "All three kernels are dispatched 1-D, one thread per element. "
            "lj_clear_cells: M^3 threads (gid >= M3 early-exits). "
            "lj_build_cells / lj_step: N threads (i >= N early-exits). Each "
            "thread MUST handle exactly one element; the host will not "
            "shrink the dispatch if you process multiple elements per "
            "thread. All buffers use MTLResourceStorageModeShared (Apple "
            "Silicon unified memory). cell_count is read via atomics from "
            "lj_clear_cells / lj_build_cells and as a plain uint* in "
            "lj_step (no atomicity required for the read-only pass)."
        ),
        kernel_names=["lj_clear_cells", "lj_build_cells", "lj_step"],
        seed_path=_SEED,
        sizes=[
            # K^3 = N; spacing = L/K; cell_size = L/M >= rcut required.
            TaskSize("N1728_M5_steps20",
                     {"N": 1728,  "M": 5,  "L": 13.8, "n_steps": 20}),
            TaskSize("N4096_M7_steps15",
                     {"N": 4096,  "M": 7,  "L": 18.4, "n_steps": 15}),
            TaskSize("N10648_M10_steps10",
                     {"N": 10648, "M": 10, "L": 25.3, "n_steps": 10}),
        ],
        held_out_sizes=[
            TaskSize("N2744_M6_steps12",
                     {"N": 2744, "M": 6, "L": 16.1, "n_steps": 12}),
        ],
    )

    def evaluate_size(self, harness, pipelines, size, chip, n_warmup, n_measure):
        N = int(size.params["N"])
        M = int(size.params["M"])
        L = float(size.params["L"])
        n_steps = int(size.params["n_steps"])
        dt = 0.005
        rcut = 2.5
        rcut2 = rcut * rcut

        cell_size = L / M
        if cell_size < rcut:
            raise ValueError(
                f"size {size.label}: cell_size={cell_size:.3f} < rcut={rcut} "
                f"(L/M must be >= rcut for the 27-neighbour scheme)"
            )
        if M < 3:
            raise ValueError(
                f"size {size.label}: M={M} < 3 (need >=3 cells/side; "
                f"otherwise periodic neighbour offsets alias)"
            )

        pos3_0, vel3_0, pos4_0, vel4_0 = _make_init(N, L)

        # --- Buffers ----------------------------------------------------
        b_posA = harness.buf_from_np(pos4_0)
        b_posB = harness.buf_zeros(pos4_0.nbytes)
        b_velA = harness.buf_from_np(vel4_0)
        b_velB = harness.buf_zeros(vel4_0.nbytes)

        M3 = M * M * M
        b_cell_count = harness.buf_from_np(np.zeros(M3, dtype=np.uint32))
        b_cell_list = harness.buf_from_np(
            np.zeros(M3 * _MAX_PER_CELL, dtype=np.uint32)
        )

        b_N    = harness.buf_scalar(N, np.uint32)
        b_M    = harness.buf_scalar(M, np.uint32)
        b_M3   = harness.buf_scalar(M3, np.uint32)
        b_L    = harness.buf_scalar(L, np.float32)
        b_rcut2 = harness.buf_scalar(rcut2, np.float32)
        b_dt   = harness.buf_scalar(dt, np.float32)
        b_MAX  = harness.buf_scalar(_MAX_PER_CELL, np.uint32)

        # --- Pipelines and threadgroup geometries -----------------------
        pso_clear = pipelines["lj_clear_cells"]
        pso_build = pipelines["lj_build_cells"]
        pso_step  = pipelines["lj_step"]
        tg_clear = min(64,  int(pso_clear.maxTotalThreadsPerThreadgroup()))
        tg_build = min(128, int(pso_build.maxTotalThreadsPerThreadgroup()))
        tg_step  = min(128, int(pso_step .maxTotalThreadsPerThreadgroup()))
        grid_clear = ((M3 + tg_clear - 1) // tg_clear) * tg_clear
        grid_build = ((N  + tg_build - 1) // tg_build) * tg_build
        grid_step  = ((N  + tg_step  - 1) // tg_step ) * tg_step

        # --- numpy views aliasing the unified-memory buffers ------------
        view_posA = harness.np_view(b_posA, np.float32, N * 4)
        view_velA = harness.np_view(b_velA, np.float32, N * 4)
        view_posB = harness.np_view(b_posB, np.float32, N * 4)
        view_velB = harness.np_view(b_velB, np.float32, N * 4)
        view_cell_count = harness.np_view(b_cell_count, np.uint32, M3)
        view_cell_list  = harness.np_view(
            b_cell_list, np.uint32, M3 * _MAX_PER_CELL,
        )

        def reset():
            view_posA[:] = pos4_0.ravel()
            view_velA[:] = vel4_0.ravel()
            view_posB[:] = 0.0
            view_velB[:] = 0.0
            view_cell_count[:] = 0
            view_cell_list[:] = 0

        def dispatch(enc):
            for step in range(n_steps):
                if step % 2 == 0:
                    pin, pout = b_posA, b_posB
                    vin, vout = b_velA, b_velB
                else:
                    pin, pout = b_posB, b_posA
                    vin, vout = b_velB, b_velA

                # 1. Clear cell counts (also clears cell_list logically by
                # making every count = 0 before the next build).
                enc.setComputePipelineState_(pso_clear)
                enc.setBuffer_offset_atIndex_(b_cell_count, 0, 0)
                enc.setBuffer_offset_atIndex_(b_M3, 0, 1)
                enc.dispatchThreads_threadsPerThreadgroup_(
                    Metal.MTLSizeMake(grid_clear, 1, 1),
                    Metal.MTLSizeMake(tg_clear, 1, 1),
                )

                # 2. Build cells from pin.
                enc.setComputePipelineState_(pso_build)
                enc.setBuffer_offset_atIndex_(pin, 0, 0)
                enc.setBuffer_offset_atIndex_(b_cell_count, 0, 1)
                enc.setBuffer_offset_atIndex_(b_cell_list, 0, 2)
                enc.setBuffer_offset_atIndex_(b_N, 0, 3)
                enc.setBuffer_offset_atIndex_(b_M, 0, 4)
                enc.setBuffer_offset_atIndex_(b_L, 0, 5)
                enc.setBuffer_offset_atIndex_(b_MAX, 0, 6)
                enc.dispatchThreads_threadsPerThreadgroup_(
                    Metal.MTLSizeMake(grid_build, 1, 1),
                    Metal.MTLSizeMake(tg_build, 1, 1),
                )

                # 3. LJ force + integrate.
                enc.setComputePipelineState_(pso_step)
                enc.setBuffer_offset_atIndex_(pin, 0, 0)
                enc.setBuffer_offset_atIndex_(pout, 0, 1)
                enc.setBuffer_offset_atIndex_(vin, 0, 2)
                enc.setBuffer_offset_atIndex_(vout, 0, 3)
                enc.setBuffer_offset_atIndex_(b_cell_count, 0, 4)
                enc.setBuffer_offset_atIndex_(b_cell_list, 0, 5)
                enc.setBuffer_offset_atIndex_(b_N, 0, 6)
                enc.setBuffer_offset_atIndex_(b_M, 0, 7)
                enc.setBuffer_offset_atIndex_(b_L, 0, 8)
                enc.setBuffer_offset_atIndex_(b_rcut2, 0, 9)
                enc.setBuffer_offset_atIndex_(b_dt, 0, 10)
                enc.setBuffer_offset_atIndex_(b_MAX, 0, 11)
                enc.dispatchThreads_threadsPerThreadgroup_(
                    Metal.MTLSizeMake(grid_step, 1, 1),
                    Metal.MTLSizeMake(tg_step, 1, 1),
                )

        # --- Warmup + measure -------------------------------------------
        for _ in range(n_warmup):
            reset()
            harness.time_dispatch(dispatch)
        samples = []
        for _ in range(n_measure):
            reset()
            samples.append(harness.time_dispatch(dispatch))
        gpu_s = float(np.median(samples))

        # --- Cell-occupancy invariant check (one-shot) ------------------
        # If MAX_PER_CELL is exceeded, build_cells silently drops particles
        # and the GPU disagrees with the CPU on which pairs are visible.
        # Read back the post-final-step cell_count and surface a clear
        # error rather than a confusing tolerance failure.
        max_occupancy = int(view_cell_count.max())
        if max_occupancy > _MAX_PER_CELL:
            return SizeResult(
                size_label=size.label, correct=False,
                error_value=float(max_occupancy), error_kind="cell_overflow",
                gpu_seconds=gpu_s, achieved=0.0, achieved_unit="GFLOPS",
                ceiling=float(chip.peak_fp32_gflops), ceiling_unit="GFLOPS",
                fraction_of_ceiling=0.0,
                extra={"max_per_cell": _MAX_PER_CELL,
                       "observed_max_occupancy": max_occupancy,
                       "N": N, "M": M, "L": L, "n_steps": n_steps},
            )

        # --- Final correctness pass -------------------------------------
        reset()
        harness.time_dispatch(dispatch)
        final_pos_view = view_posA if n_steps % 2 == 0 else view_posB
        got_pos = final_pos_view.copy().reshape(N, 4)[:, :3]

        ref_pos, _ = _cpu_reference(pos3_0, vel3_0, L, rcut, dt, n_steps)
        max_pos = float(np.max(np.abs(ref_pos)))
        # Both CPU and GPU integrate WITHOUT wrapping, so unwrapped
        # positions agree up to fp32 ordering noise. Empirically that's
        # ~1e-6 absolute over the n_steps used; 1e-4 leaves 100x margin
        # for legitimate candidates while a no-op or sign-flipped force
        # produces a displacement disagreement >> 1e-2 (advection from
        # the small initial velocities, plus force-driven motion).
        # Specifically NO scaling on |max_pos|: the scale of the
        # disagreement is set by the trajectory step, not the box.
        err = float(np.max(np.abs(got_pos - ref_pos)))
        tol = 1e-4
        correct = err <= tol

        # --- Throughput metric ------------------------------------------
        # Useful work: each unordered pair within rcut is visited from
        # BOTH endpoints in lj_step. Use 20 FLOPs / pair-visit (matching
        # nbody's accounting so the % numbers are mentally comparable).
        # The cell-list build/clear are book-keeping; not counted here.
        pair_count = _count_pairs_within_rcut(pos3_0, L, rcut)
        flops_per_step = 2.0 * pair_count * 20.0
        total_flops = flops_per_step * n_steps
        achieved = gflops(total_flops, gpu_s)
        ceiling = float(chip.peak_fp32_gflops)
        return SizeResult(
            size_label=size.label,
            correct=correct,
            error_value=err,
            error_kind="max_abs_pos",
            gpu_seconds=gpu_s,
            achieved=achieved,
            achieved_unit="GFLOPS (useful pairs only)",
            ceiling=ceiling,
            ceiling_unit="GFLOPS",
            fraction_of_ceiling=achieved / ceiling if ceiling > 0 else 0.0,
            extra={"tol": tol, "N": N, "M": M, "L": L, "n_steps": n_steps,
                   "dt": dt, "rcut": rcut, "pair_count": pair_count,
                   "max_cell_occupancy": max_occupancy},
        )
