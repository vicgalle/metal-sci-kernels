"""3D acoustic wave equation: 7-point spatial Laplacian + leapfrog (Plan §1, Task 3).

Second-order finite difference in time and space:

    u^{n+1} = 2 u^n - u^{n-1} + alpha * Laplacian(u^n)

with the 3D 7-point isotropic Laplacian and Dirichlet boundary conditions.
``alpha = (c dt / dx)^2`` and 3D CFL stability requires ``alpha < 1/3``;
we use 0.18 for a comfortable margin.

The candidate writes one ``wave_step`` kernel that performs ONE timestep.
The host triple-buffers (u_prev, u_curr, u_next) across ``n_steps``,
rotating buffer bindings each call; all dispatches share one command
buffer so the GPU timing covers the whole multi-step run.

Memory traffic per step at the cache-perfect limit is 12 B/cell (load
u_prev + load u_curr + store u_next; the 6 stencil neighbours are
amortised by L1/L2 reuse). The roofline is BW-bound at the chip's peak
DRAM bandwidth, and the plan's 2.5D-blocking target sits at ~50-70% of
peak.
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


_SEED = Path(__file__).resolve().parent.parent.parent / "seeds" / "wave3d.metal"


def _make_init(nx: int, ny: int, nz: int) -> np.ndarray:
    """Smooth 3D Gaussian bump centred in the box, zero on boundary, fp32.

    Shape (NZ, NY, NX), C-contiguous, peak amplitude 1.0. Returning a
    Gaussian (rather than a delta or random field) keeps the wavefront
    smooth and well-resolved by a 7-point stencil — important for
    correctness checking where high-frequency content would amplify the
    fp32 reordering noise across CPU and GPU.
    """
    iz = np.arange(nz, dtype=np.float32).reshape(nz, 1, 1)
    iy = np.arange(ny, dtype=np.float32).reshape(1, ny, 1)
    ix = np.arange(nx, dtype=np.float32).reshape(1, 1, nx)
    cx = np.float32((nx - 1) / 2.0)
    cy = np.float32((ny - 1) / 2.0)
    cz = np.float32((nz - 1) / 2.0)
    sigma = np.float32(0.1 * min(nx, ny, nz))
    r2 = (ix - cx) ** 2 + (iy - cy) ** 2 + (iz - cz) ** 2
    u = np.exp(-r2 / (np.float32(2.0) * sigma * sigma)).astype(np.float32)
    # Hard-zero the boundary so Dirichlet BC is exactly preserved.
    u[0, :, :] = 0.0
    u[-1, :, :] = 0.0
    u[:, 0, :] = 0.0
    u[:, -1, :] = 0.0
    u[:, :, 0] = 0.0
    u[:, :, -1] = 0.0
    return np.ascontiguousarray(u)


def _cpu_reference(u0: np.ndarray, alpha: float, n_steps: int) -> np.ndarray:
    """Reference 3D leapfrog wave update, fp32, vectorized via numpy slicing.

    Mirrors the GPU kernel: at boundary, u_next = u_curr. Returns u^{n_steps}.
    Uses the same 3-buffer rotation as the host so the buffer that holds
    the result after `n_steps` steps matches the GPU side.
    """
    a = np.float32(alpha)
    two = np.float32(2.0)
    six = np.float32(6.0)
    u_prev = u0.astype(np.float32, copy=True)
    u_curr = u0.astype(np.float32, copy=True)
    u_next = np.zeros_like(u0)
    for _ in range(n_steps):
        # Interior (1:-1 each axis): 7-point Laplacian + leapfrog.
        c = u_curr[1:-1, 1:-1, 1:-1]
        u_next[1:-1, 1:-1, 1:-1] = (
            two * c - u_prev[1:-1, 1:-1, 1:-1]
            + a * (
                u_curr[1:-1, 1:-1, :-2] + u_curr[1:-1, 1:-1, 2:]
                + u_curr[1:-1, :-2, 1:-1] + u_curr[1:-1, 2:, 1:-1]
                + u_curr[:-2, 1:-1, 1:-1] + u_curr[2:, 1:-1, 1:-1]
                - six * c
            )
        )
        # Boundary copy (Dirichlet stays put).
        u_next[0, :, :] = u_curr[0, :, :]
        u_next[-1, :, :] = u_curr[-1, :, :]
        u_next[:, 0, :] = u_curr[:, 0, :]
        u_next[:, -1, :] = u_curr[:, -1, :]
        u_next[:, :, 0] = u_curr[:, :, 0]
        u_next[:, :, -1] = u_curr[:, :, -1]
        # Rotate (prev, curr, next) -> (curr, next, prev), matching host.
        u_prev, u_curr, u_next = u_curr, u_next, u_prev
    return u_curr


@register_task("wave3d")
class Wave3DTask(Task):
    spec = TaskSpec(
        name="wave3d",
        description=(
            "3D acoustic wave equation with a 7-point spatial Laplacian and "
            "second-order leapfrog time integration:\n"
            "  u_next[i,j,k] = 2 u_curr[i,j,k] - u_prev[i,j,k]\n"
            "                + alpha * ( u_curr[i-1,j,k] + u_curr[i+1,j,k]\n"
            "                          + u_curr[i,j-1,k] + u_curr[i,j+1,k]\n"
            "                          + u_curr[i,j,k-1] + u_curr[i,j,k+1]\n"
            "                          - 6 u_curr[i,j,k] )\n"
            "alpha = (c * dt / dx)^2; the host uses alpha = 0.18, comfortably "
            "below the 3D CFL limit of 1/3. Dirichlet BC: every face cell "
            "(i==0, j==0, k==0, i==NX-1, j==NY-1, k==NZ-1) MUST copy "
            "u_curr -> u_next unchanged.\n\n"
            "Storage is row-major float32 of shape (NZ, NY, NX) — i is the "
            "fast (x) axis, j the middle (y) axis, k the slow (z) axis. "
            "Linear index: idx = (k * NY + j) * NX + i. The host triple-"
            "buffers across n_steps, rotating (prev, curr, next) bindings "
            "each call; all dispatches share one command buffer for "
            "accurate end-to-end GPU timing. Initial state has u_prev = "
            "u_curr (zero initial velocity in time)."
        ),
        kernel_signatures=(
            "kernel void wave_step(device const float *u_prev [[buffer(0)]],\n"
            "                      device const float *u_curr [[buffer(1)]],\n"
            "                      device       float *u_next [[buffer(2)]],\n"
            "                      constant uint      &NX     [[buffer(3)]],\n"
            "                      constant uint      &NY     [[buffer(4)]],\n"
            "                      constant uint      &NZ     [[buffer(5)]],\n"
            "                      constant float     &alpha  [[buffer(6)]],\n"
            "                      uint3 gid [[thread_position_in_grid]]);\n"
            "\n"
            "Grid is dispatched 3-D as `threadsPerGrid = (NX, NY, NZ)`, one "
            "thread per output cell — guard with `if (i >= NX || j >= NY "
            "|| k >= NZ) return;`. Each thread MUST update exactly one "
            "cell; the host will not shrink the dispatch if you process "
            "multiple cells per thread, so extra threads just idle. "
            "Boundary cells (i==0, j==0, k==0, i==NX-1, j==NY-1, k==NZ-1) "
            "MUST copy u_curr -> u_next unchanged. Threadgroup-memory "
            "tiling and 2.5D blocking (one YX tile in shared memory, "
            "marching through Z while keeping a small Z window in "
            "registers) are the canonical optimizations for this kernel."
        ),
        kernel_names=["wave_step"],
        seed_path=_SEED,
        sizes=[
            # Three regimes: 64^3 fits comfortably in L2/SLC and is launch-
            # overhead-bound; 160^3 (~46 MB working set) spills the M1 Pro
            # 24 MB SLC; 192^3 (~80 MB) is solidly DRAM-bound.
            TaskSize("64x64x64_30",    {"nx": 64,  "ny": 64,  "nz": 64,  "n_steps": 30}),
            TaskSize("160x160x160_20", {"nx": 160, "ny": 160, "nz": 160, "n_steps": 20}),
            TaskSize("192x192x192_15", {"nx": 192, "ny": 192, "nz": 192, "n_steps": 15}),
        ],
        held_out_sizes=[
            TaskSize("128x128x128_20", {"nx": 128, "ny": 128, "nz": 128, "n_steps": 20}),
        ],
    )

    def evaluate_size(self, harness, pipelines, size, chip, n_warmup, n_measure):
        nx = int(size.params["nx"])
        ny = int(size.params["ny"])
        nz = int(size.params["nz"])
        n_steps = int(size.params["n_steps"])
        alpha = 0.18

        u0 = _make_init(nx, ny, nz)            # (NZ, NY, NX) fp32
        nbytes = u0.nbytes

        # Triple-buffer: A holds initial u_prev, B holds initial u_curr (=u0),
        # C is scratch (overwritten in step 0). After step k, u^{k+1} sits
        # in buffer (k + 2) % 3; thus after `n_steps` steps the answer is
        # in buffer (n_steps + 1) % 3.
        bA = harness.buf_from_np(u0)
        bB = harness.buf_from_np(u0)
        bC = harness.buf_zeros(nbytes)
        bNX = harness.buf_scalar(nx, np.uint32)
        bNY = harness.buf_scalar(ny, np.uint32)
        bNZ = harness.buf_scalar(nz, np.uint32)
        bAlpha = harness.buf_scalar(alpha, np.float32)

        pso = pipelines["wave_step"]
        max_tg = int(pso.maxTotalThreadsPerThreadgroup())
        # 8x8x4 = 256 threads is a robust 3D default. Scale down on the
        # rare PSO with a tighter cap (high register pressure variants).
        tg_x, tg_y, tg_z = 8, 8, 4
        while tg_x * tg_y * tg_z > max_tg:
            if tg_z > 1:
                tg_z //= 2
            elif tg_y > 1:
                tg_y //= 2
            else:
                tg_x //= 2
        grid_x = ((nx + tg_x - 1) // tg_x) * tg_x
        grid_y = ((ny + tg_y - 1) // tg_y) * tg_y
        grid_z = ((nz + tg_z - 1) // tg_z) * tg_z

        view_A = harness.np_view(bA, np.float32, nx * ny * nz)
        view_B = harness.np_view(bB, np.float32, nx * ny * nz)
        view_C = harness.np_view(bC, np.float32, nx * ny * nz)
        views = [view_A, view_B, view_C]
        bufs  = [bA, bB, bC]

        def reset():
            view_A[:] = u0.ravel()  # initial u_prev
            view_B[:] = u0.ravel()  # initial u_curr (== u_prev: zero initial velocity)
            view_C[:] = 0.0         # scratch; will be fully overwritten in step 0

        def dispatch(enc):
            enc.setComputePipelineState_(pso)
            enc.setBuffer_offset_atIndex_(bNX, 0, 3)
            enc.setBuffer_offset_atIndex_(bNY, 0, 4)
            enc.setBuffer_offset_atIndex_(bNZ, 0, 5)
            enc.setBuffer_offset_atIndex_(bAlpha, 0, 6)
            for step in range(n_steps):
                prev_buf = bufs[step % 3]
                curr_buf = bufs[(step + 1) % 3]
                next_buf = bufs[(step + 2) % 3]
                enc.setBuffer_offset_atIndex_(prev_buf, 0, 0)
                enc.setBuffer_offset_atIndex_(curr_buf, 0, 1)
                enc.setBuffer_offset_atIndex_(next_buf, 0, 2)
                enc.dispatchThreads_threadsPerThreadgroup_(
                    Metal.MTLSizeMake(grid_x, grid_y, grid_z),
                    Metal.MTLSizeMake(tg_x, tg_y, tg_z),
                )

        # Warmup
        for _ in range(n_warmup):
            reset()
            harness.time_dispatch(dispatch)
        # Measured
        samples = []
        for _ in range(n_measure):
            reset()
            samples.append(harness.time_dispatch(dispatch))
        gpu_s = float(np.median(samples))

        # Final correctness pass: re-run from a clean reset so the buffer
        # holding the answer is deterministic w.r.t. the ping-pong index.
        reset()
        harness.time_dispatch(dispatch)
        final_view = views[(n_steps + 1) % 3]
        got = final_view.copy().reshape(nz, ny, nx)

        expected = _cpu_reference(u0, alpha, n_steps)
        max_ref = float(np.max(np.abs(expected)))
        err = float(np.max(np.abs(got - expected)))
        # fp32 stencil over n_steps accumulates ~O(eps * n_steps * max|u|)
        # rounding noise; 1e-4 absolute is comfortably above the observed
        # CPU/GPU drift without masking real bugs (a no-op or sign-flipped
        # update produces O(0.1) disagreement once the wave propagates).
        tol = 1e-4 + 1e-5 * max_ref
        correct = err <= tol

        # BW-bound roofline. At the cache-perfect limit each cell touches
        # DRAM exactly three times per step (load u_prev, load u_curr,
        # store u_next); the 6 stencil neighbours of u_curr are amortised
        # by L1/L2 reuse. 12 B/cell is the optimistic ceiling, matching
        # the convention used in the heat2d task (8 B/cell there, since
        # heat2d only has 2 buffers in the ping-pong).
        bytes_per_step = 12.0 * nx * ny * nz
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
            achieved_unit="GB/s (effective, 12 B/cell)",
            ceiling=ceiling,
            ceiling_unit="GB/s",
            fraction_of_ceiling=achieved / ceiling if ceiling > 0 else 0.0,
            extra={"tol": tol, "nx": nx, "ny": ny, "nz": nz,
                   "n_steps": n_steps, "alpha": alpha},
        )
