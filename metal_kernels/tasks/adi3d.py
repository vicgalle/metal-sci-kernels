"""3D LOD-ADI heat equation: three tridiagonal solves per timestep
(R7 wavefront / strict-dependency sweeps).

One timestep of the Locally-One-Dimensional (LOD) ADI scheme for the
heat equation u_t = nabla^2 u factors the implicit update into three
sequential 1-D solves, one per Cartesian axis:

    (I - mu * Dxx) v1       = u^n          (x-sweep)
    (I - mu * Dyy) v2       = v1           (y-sweep)
    (I - mu * Dzz) u^{n+1}  = v2           (z-sweep)

where ``mu = dt / h^2`` (host uses ``mu = 0.5``; LOD-ADI is
unconditionally stable so there is no CFL constraint). Each per-line
system has the constant tridiagonal coefficients ``(-mu, 1+2mu, -mu)``
with Dirichlet endpoints (the line's two boundary cells, untouched).
Cube-face Dirichlet is preserved across the whole timestep: every cell
with x in {0, NX-1}, y in {0, NY-1}, or z in {0, NZ-1} keeps its
initial value through all three sub-steps.

Optimization lever (uniquely stressed by this task; see FUTURE_TASKS R7):
the same data must be stride-1 along x, then y, then z within one
timestep, while every per-line solve has a strict serial dependence.
The seed dispatches one thread per line and streams the modified RHS
through device memory (forward sweep writes d'_i to u_out, backward
sweep reads it back), spending ~2x the cache-perfect DRAM traffic
(16 B/cell/sweep vs 8 B/cell/sweep optimum). The candidate's choices:

  * Hold the per-thread line in threadgroup memory (W threads x N floats
    per TG) so forward+backward stay local; one DRAM read and one DRAM
    write per cell per sweep.
  * Parallel cyclic reduction (PCR) or cyclic reduction (CR) within a
    line, parallelizing the serial Thomas in ``log2(N)`` rounds at the
    cost of redundant arithmetic.
  * Sweep fusion / transposed-layout management: fuse x and y (or all
    three) in one kernel that keeps a slab in TG memory and transposes
    between sweeps so the active axis is always stride-1 locally.

Held-out (NX=256, NY=192, NZ=128): a non-cubic prism that (i) is
solidly DRAM-bound (~50 MB ping-pong working set, well past M1 Pro's
24 MB SLC, so the bandwidth lever actually matters here -- mirrors
fft3d's 256^3 held-out) and (ii) catches the canonical "I assumed
N x N x N" overfit -- a candidate that hardcodes ``N = NX`` for the
y-sweep, or pads x/y/z to a single TG width, will fail correctness
the moment the axes disagree.

Roofline: BW-bound at 24 B/cell/step (three sweeps, each at the
cache-perfect 8 B/cell limit). The smallest in-distribution size 64^3
is SLC-resident on M1 Pro, so its reported fraction-of-ceiling may
exceed 1 -- same caveat as fft3d 32^3 and lbm 256^2.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import Metal

from ..harness import MetalHarness
from ..hardware import ChipSpec
from ..task import (
    Task, TaskSize, TaskSpec, SizeResult,
    gb_per_s, register_task,
)


_SEED = Path(__file__).resolve().parent.parent.parent / "seeds" / "adi3d.metal"


def _make_init(NX: int, NY: int, NZ: int) -> np.ndarray:
    """Smooth 3D Gaussian bump centred in the box, zero on every face.

    Shape (NZ, NY, NX) fp32, C-contiguous. A Gaussian (rather than a
    delta or random field) keeps high-frequency content low so fp32
    accumulation in the Thomas chain stays well-behaved across many
    timesteps; sigma is anchored to the geometric mean of the three
    axis lengths so the bump scales sensibly on non-cubic prisms.
    """
    iz = np.arange(NZ, dtype=np.float32).reshape(NZ, 1, 1)
    iy = np.arange(NY, dtype=np.float32).reshape(1, NY, 1)
    ix = np.arange(NX, dtype=np.float32).reshape(1, 1, NX)
    cx = np.float32((NX - 1) / 2.0)
    cy = np.float32((NY - 1) / 2.0)
    cz = np.float32((NZ - 1) / 2.0)
    # Geometric mean of axis lengths keeps the bump well-resolved on
    # prisms as well as cubes.
    L = float((NX * NY * NZ) ** (1.0 / 3.0))
    sigma = np.float32(0.15 * L)
    r2 = (ix - cx) ** 2 + (iy - cy) ** 2 + (iz - cz) ** 2
    u = np.exp(-r2 / (np.float32(2.0) * sigma * sigma)).astype(np.float32)
    # Hard-zero the cube faces so Dirichlet BC is preserved exactly.
    u[0, :, :] = 0.0
    u[-1, :, :] = 0.0
    u[:, 0, :] = 0.0
    u[:, -1, :] = 0.0
    u[:, :, 0] = 0.0
    u[:, :, -1] = 0.0
    return np.ascontiguousarray(u)


def _thomas_constant_coef_last_axis(rhs: np.ndarray, mu: float) -> np.ndarray:
    """Vectorised constant-coefficient Thomas along the last axis.

    The system on every length-N line is::

        -mu v_{i-1} + (1+2mu) v_i + -mu v_{i+1} = rhs_i,   1 <= i <= N-2

    with Dirichlet endpoints ``v_0 = rhs_0`` and ``v_{N-1} = rhs_{N-1}``
    (rhs[..., 0] and rhs[..., -1] are interpreted as the boundary
    sources, untouched in the output). ``rhs`` may have any leading
    shape; the solve broadcasts over leading axes.
    """
    rhs = np.ascontiguousarray(rhs, dtype=np.float32)
    out = rhs.copy()
    N = rhs.shape[-1]
    if N < 3:
        return out

    a = np.float32(-mu)
    b = np.float32(1.0 + 2.0 * mu)
    c = np.float32(-mu)

    # cprime is 1-D (line-independent for constant a,b,c); dprime carries
    # the full leading shape.
    cprime = np.zeros(N, dtype=np.float32)
    dprime = np.zeros_like(rhs, dtype=np.float32)
    bd_lo = rhs[..., 0]
    bd_hi = rhs[..., -1]

    # i = 1 (low-side Dirichlet correction r_1 += mu * bd_lo).
    cprime[1] = c / b
    dprime[..., 1] = (rhs[..., 1] + np.float32(mu) * bd_lo) / b
    # i = 2 .. N - 3.
    for i in range(2, N - 2):
        denom = b - a * cprime[i - 1]
        cprime[i] = c / denom
        dprime[..., i] = (rhs[..., i] - a * dprime[..., i - 1]) / denom
    # i = N - 2 (hi-side Dirichlet correction r_{N-2} += mu * bd_hi).
    i = N - 2
    denom = b - a * cprime[i - 1]
    cprime[i] = c / denom
    dprime[..., i] = (
        (rhs[..., i] + np.float32(mu) * bd_hi) - a * dprime[..., i - 1]
    ) / denom

    # Backward sub.
    out[..., N - 2] = dprime[..., N - 2]
    for i in range(N - 3, 0, -1):
        out[..., i] = dprime[..., i] - cprime[i] * out[..., i + 1]
    # out[..., 0] and out[..., -1] keep their rhs values from .copy().
    return out


def _sweep_along_axis(u: np.ndarray, mu: float, axis: int) -> np.ndarray:
    """One LOD-ADI sweep along ``axis`` of ``u`` (axis 0=z, 1=y, 2=x).

    Lines whose OFF-axis indices both sit strictly interior to the cube
    get a Thomas solve along the active axis. Lines that touch any cube
    face in their off-axis indices copy through unchanged -- this keeps
    every cube-face cell at its initial value across the whole timestep.
    """
    u_t = np.moveaxis(u, axis, -1).copy()        # (other1, other2, N_active)
    out_t = u_t.copy()
    if u_t.shape[0] >= 3 and u_t.shape[1] >= 3:
        interior = u_t[1:-1, 1:-1, :]
        solved = _thomas_constant_coef_last_axis(interior, mu)
        out_t[1:-1, 1:-1, :] = solved
    return np.moveaxis(out_t, -1, axis).copy()


def _cpu_reference(u0: np.ndarray, mu: float, n_steps: int) -> np.ndarray:
    """Reference 3D LOD-ADI heat update, fp32, n_steps timesteps."""
    u = u0.astype(np.float32, copy=True)
    for _ in range(n_steps):
        u = _sweep_along_axis(u, mu, axis=2)     # x-sweep (axis 2 is x, fast)
        u = _sweep_along_axis(u, mu, axis=1)     # y-sweep
        u = _sweep_along_axis(u, mu, axis=0)     # z-sweep
    return u


@register_task("adi3d")
class ADI3DTask(Task):
    spec = TaskSpec(
        name="adi3d",
        description=(
            "3D Locally-One-Dimensional (LOD) ADI for the heat equation. "
            "One timestep solves three constant-coefficient tridiagonal "
            "systems sequentially along x, then y, then z:\n"
            "  (I - mu * Dxx) v1      = u^n         (x-sweep)\n"
            "  (I - mu * Dyy) v2      = v1          (y-sweep)\n"
            "  (I - mu * Dzz) u^{n+1} = v2          (z-sweep)\n"
            "where mu = dt/h^2 (host uses mu = 0.5; LOD-ADI is "
            "unconditionally stable, no CFL). Each per-line system has "
            "constant tridiagonal entries\n"
            "  -mu * v_{i-1} + (1 + 2 mu) * v_i + -mu * v_{i+1} = rhs_i,  1 <= i <= N-2\n"
            "with Dirichlet endpoints v_0 = rhs_0, v_{N-1} = rhs_{N-1} "
            "(the line's two boundary cells, untouched by the solve).\n\n"
            "Cube-face Dirichlet: every cell with i in {0, NX-1} OR j in "
            "{0, NY-1} OR k in {0, NZ-1} (any cube face) MUST stay at "
            "its initial value across the entire timestep. The harness "
            "enforces this convention: per sweep, lines whose two "
            "OFF-axis indices both sit strictly interior on the cube "
            "get a Thomas solve along the active axis; lines that touch "
            "a cube face in their off-axis indices copy u_in -> u_out "
            "unchanged. The result is that all six cube faces are "
            "preserved through every sub-step.\n\n"
            "Storage is row-major float32 of shape (NZ, NY, NX) with i "
            "the fast (x) axis, j the middle (y) axis, k the slow (z) "
            "axis. Linear index: idx = (k * NY + j) * NX + i. NX, NY, "
            "and NZ are independent positive integers and need not be "
            "equal. The host calls three separate kernels -- adi_x, "
            "adi_y, adi_z -- in that order, ping-ponging two device "
            "buffers, with all dispatches sharing one command buffer "
            "for accurate end-to-end GPU timing of the n_steps run."
        ),
        kernel_signatures=(
            "kernel void adi_x(device const float *u_in   [[buffer(0)]],\n"
            "                  device       float *u_out  [[buffer(1)]],\n"
            "                  constant uint      &NX     [[buffer(2)]],\n"
            "                  constant uint      &NY     [[buffer(3)]],\n"
            "                  constant uint      &NZ     [[buffer(4)]],\n"
            "                  constant float     &mu     [[buffer(5)]],\n"
            "                  uint2 gid [[thread_position_in_grid]]);\n"
            "kernel void adi_y(device const float *u_in   [[buffer(0)]],\n"
            "                  device       float *u_out  [[buffer(1)]],\n"
            "                  constant uint      &NX     [[buffer(2)]],\n"
            "                  constant uint      &NY     [[buffer(3)]],\n"
            "                  constant uint      &NZ     [[buffer(4)]],\n"
            "                  constant float     &mu     [[buffer(5)]],\n"
            "                  uint2 gid [[thread_position_in_grid]]);\n"
            "kernel void adi_z(device const float *u_in   [[buffer(0)]],\n"
            "                  device       float *u_out  [[buffer(1)]],\n"
            "                  constant uint      &NX     [[buffer(2)]],\n"
            "                  constant uint      &NY     [[buffer(3)]],\n"
            "                  constant uint      &NZ     [[buffer(4)]],\n"
            "                  constant float     &mu     [[buffer(5)]],\n"
            "                  uint2 gid [[thread_position_in_grid]]);\n"
            "\n"
            "Dispatch geometry (host-fixed; identical pattern across the "
            "three kernels, with the two off-axis indices on gid.x and "
            "gid.y):\n"
            "  adi_x: threadsPerGrid = (NY, NZ, 1), TG = (32, 1, 1).\n"
            "         gid.x = j (off-axis y), gid.y = k (off-axis z).\n"
            "  adi_y: threadsPerGrid = (NX, NZ, 1), TG = (32, 1, 1).\n"
            "         gid.x = i (off-axis x), gid.y = k (off-axis z).\n"
            "  adi_z: threadsPerGrid = (NX, NY, 1), TG = (32, 1, 1).\n"
            "         gid.x = i (off-axis x), gid.y = j (off-axis y).\n"
            "Convention: one thread owns one full Thomas line along the "
            "active axis. Each thread MUST early-exit if its gid is past "
            "the corresponding axis length. Boundary lines (those whose "
            "off-axis indices touch a cube face) MUST copy u_in -> u_out "
            "cell-by-cell.\n\n"
            "If you cap the threadgroup with [[max_total_threads_per_"
            "threadgroup(W)]], place the attribute on the kernel "
            "declaration line itself, and remember the host dispatches "
            "TG = (32, 1, 1); a cap below 32 will be rejected. Buffers "
            "0 and 1 are read/write and ping-ponged across timesteps, "
            "so do NOT assume u_in and u_out alias fixed addresses. "
            "The host calls adi_x -> adi_y -> adi_z back-to-back per "
            "timestep, with the output of one sweep being the input of "
            "the next; n_steps total timesteps share one command buffer."
        ),
        kernel_names=["adi_x", "adi_y", "adi_z"],
        seed_path=_SEED,
        sizes=[
            # 64^3 = 1 MB per buffer (L2/SLC-resident, launch-bound regime).
            # 96^3 ~ 3.5 MB per buffer (SLC).
            # 128^3 ~ 8 MB per buffer (16 MB across ping-pong fits the
            # 24 MB M1 Pro SLC).
            TaskSize("N64_20",  {"NX": 64,  "NY": 64,  "NZ": 64,  "n_steps": 20}),
            TaskSize("N96_15",  {"NX": 96,  "NY": 96,  "NZ": 96,  "n_steps": 15}),
            TaskSize("N128_10", {"NX": 128, "NY": 128, "NZ": 128, "n_steps": 10}),
        ],
        held_out_sizes=[
            # Non-cubic prism, ~25 MB per buffer (50 MB across the ping-
            # pong) -- solidly DRAM-bound, well past M1 Pro's 24 MB SLC.
            # The three axes are all different, catching any kernel that
            # hardcoded NX = NY = NZ.
            TaskSize("P256x192x128_10", {"NX": 256, "NY": 192, "NZ": 128, "n_steps": 10}),
        ],
    )

    def evaluate_size(self, harness, pipelines, size, chip, n_warmup, n_measure):
        NX = int(size.params["NX"])
        NY = int(size.params["NY"])
        NZ = int(size.params["NZ"])
        n_steps = int(size.params["n_steps"])
        mu = 0.5

        if min(NX, NY, NZ) < 3:
            raise ValueError(
                f"adi3d: every axis must be >= 3 (got NX={NX}, NY={NY}, NZ={NZ})"
            )

        # ---- inputs ---------------------------------------------------
        u0 = _make_init(NX, NY, NZ)                # (NZ, NY, NX) fp32
        nbytes = u0.nbytes

        # ---- buffers + pipelines --------------------------------------
        bA = harness.buf_from_np(u0)
        bB = harness.buf_zeros(nbytes)
        bNX = harness.buf_scalar(NX, np.uint32)
        bNY = harness.buf_scalar(NY, np.uint32)
        bNZ = harness.buf_scalar(NZ, np.uint32)
        bMu = harness.buf_scalar(mu, np.float32)

        pso_x = pipelines["adi_x"]
        pso_y = pipelines["adi_y"]
        pso_z = pipelines["adi_z"]

        # Host dispatch is TG = (32, 1, 1). A candidate that caps the TG
        # below 32 has misread the convention; fail loudly rather than
        # silently shrinking the dispatch.
        for pso, name in [(pso_x, "adi_x"), (pso_y, "adi_y"), (pso_z, "adi_z")]:
            if int(pso.maxTotalThreadsPerThreadgroup()) < 32:
                raise RuntimeError(
                    f"{name} declares max_total_threads_per_threadgroup="
                    f"{int(pso.maxTotalThreadsPerThreadgroup())} which is "
                    f"smaller than the host-dispatched TG width 32"
                )

        tg = Metal.MTLSizeMake(32, 1, 1)
        grid_x = Metal.MTLSizeMake(NY, NZ, 1)
        grid_y = Metal.MTLSizeMake(NX, NZ, 1)
        grid_z = Metal.MTLSizeMake(NX, NY, 1)
        passes = [(pso_x, grid_x), (pso_y, grid_y), (pso_z, grid_z)]

        total_cells = NX * NY * NZ
        view_A = harness.np_view(bA, np.float32, total_cells)
        view_B = harness.np_view(bB, np.float32, total_cells)
        bufs = [bA, bB]

        def reset():
            view_A[:] = u0.ravel()
            view_B[:] = 0.0

        def dispatch(enc):
            enc.setBuffer_offset_atIndex_(bNX, 0, 2)
            enc.setBuffer_offset_atIndex_(bNY, 0, 3)
            enc.setBuffer_offset_atIndex_(bNZ, 0, 4)
            enc.setBuffer_offset_atIndex_(bMu, 0, 5)
            s = 0
            for _ in range(n_steps):
                for pso, grid in passes:
                    in_buf  = bufs[s & 1]
                    out_buf = bufs[(s + 1) & 1]
                    enc.setComputePipelineState_(pso)
                    enc.setBuffer_offset_atIndex_(in_buf,  0, 0)
                    enc.setBuffer_offset_atIndex_(out_buf, 0, 1)
                    enc.dispatchThreads_threadsPerThreadgroup_(grid, tg)
                    s += 1

        # ---- warmup + measure -----------------------------------------
        for _ in range(n_warmup):
            reset()
            harness.time_dispatch(dispatch)
        samples = []
        for _ in range(n_measure):
            reset()
            samples.append(harness.time_dispatch(dispatch))
        gpu_s = float(np.median(samples))

        # ---- correctness ----------------------------------------------
        reset()
        harness.time_dispatch(dispatch)
        # After 3*n_steps sweeps, the answer is in bufs[(3*n_steps) & 1]
        # = bufs[n_steps & 1] (3 is odd, so parity matches n_steps).
        final_view = view_A if (n_steps & 1) == 0 else view_B
        got = final_view.copy().reshape(NZ, NY, NX)

        expected = _cpu_reference(u0, mu, n_steps)
        max_ref = float(np.max(np.abs(expected)))
        err = float(np.max(np.abs(got - expected)))
        # fp32 constant-coef Thomas accumulates ~O(eps * N) error per line
        # solve; 3 sweeps per step times n_steps times max-axis Thomas of
        # length ~N gives roughly eps * 3 * n_steps * N_max in absolute
        # drift. For our (n_steps, N_max) combos that's well under 1e-3
        # against |u| ~ 1; we set tol = 1e-3 + 1e-3 * |u| to cover the
        # ratio without masking real bugs (a wrong axis index or a sign
        # flip drives error to O(0.1) once diffusion smooths the bump).
        tol = 1e-3 + 1e-3 * max_ref
        correct = err <= tol

        # ---- roofline -------------------------------------------------
        # Three sweeps per step, each at the cache-perfect 8 B/cell limit
        # (one read of u_in, one write of u_out per cell; the d' carrier
        # ideally lives in TG memory and never hits DRAM). 24 B/cell/step
        # is the optimistic ceiling. Matches the 8 B/cell convention used
        # in heat2d (single sweep) and morton (single sweep).
        bytes_per_step = 24.0 * float(total_cells)
        bytes_total = bytes_per_step * n_steps
        achieved = gb_per_s(bytes_total, gpu_s)
        ceiling = float(chip.peak_bw_gb_s)
        return SizeResult(
            size_label=size.label,
            correct=correct,
            error_value=err,
            error_kind="max_abs",
            gpu_seconds=gpu_s,
            achieved=achieved,
            achieved_unit="GB/s (effective, 24 B/cell/step across 3 sweeps)",
            ceiling=ceiling,
            ceiling_unit="GB/s",
            fraction_of_ceiling=achieved / ceiling if ceiling > 0 else 0.0,
            extra={
                "tol": tol, "NX": NX, "NY": NY, "NZ": NZ,
                "n_steps": n_steps, "mu": mu, "max_ref": max_ref,
            },
        )
