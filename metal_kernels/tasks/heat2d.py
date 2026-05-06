"""2D heat equation: 5-point stencil, multi-step ping-pong.

The candidate writes a ``heat_step`` kernel that performs ONE timestep of
the 2D heat equation. The host iterates ``n_steps`` times, ping-ponging
between two buffers; all dispatches share one command buffer for
accurate end-to-end GPU timing.

Memory traffic per step is ~8 bytes/cell (read 1 float, write 1 float;
the 4 stencil neighbours are amortised by L1/L2 cache reuse). The
roofline is BW-bound at the chip's peak DRAM bandwidth.
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


_SEED = Path(__file__).resolve().parent.parent.parent / "seeds" / "heat2d.metal"


def _cpu_reference(u0: np.ndarray, alpha: float, n_steps: int) -> np.ndarray:
    """Reference 5-point stencil with Dirichlet BC, fp32."""
    u = u0.astype(np.float32, copy=True)
    v = np.empty_like(u)
    for _ in range(n_steps):
        # Interior update (vectorized).
        v[1:-1, 1:-1] = (
            u[1:-1, 1:-1]
            + alpha * (
                u[1:-1, :-2] + u[1:-1, 2:]
                + u[:-2, 1:-1] + u[2:, 1:-1]
                - 4.0 * u[1:-1, 1:-1]
            )
        )
        # Boundary stays
        v[0, :] = u[0, :]
        v[-1, :] = u[-1, :]
        v[:, 0] = u[:, 0]
        v[:, -1] = u[:, -1]
        u, v = v, u
    return u


def _make_init(nx: int, ny: int) -> np.ndarray:
    """Smooth bump in the middle, zero on boundary, fp32, contiguous."""
    rng = np.random.default_rng(0xC0DE)
    u = rng.uniform(0.0, 1.0, size=(ny, nx)).astype(np.float32)
    # Zero the boundary so Dirichlet BC has a well-defined zero value
    u[0, :] = 0.0
    u[-1, :] = 0.0
    u[:, 0] = 0.0
    u[:, -1] = 0.0
    return np.ascontiguousarray(u)


@register_task("heat2d")
class Heat2DTask(Task):
    spec = TaskSpec(
        name="heat2d",
        description=(
            "2D heat equation with a 5-point stencil:\n"
            "  u_new[i,j] = u[i,j] + alpha * (u[i-1,j] + u[i+1,j]\n"
            "                                 + u[i,j-1] + u[i,j+1]\n"
            "                                 - 4 u[i,j])\n"
            "Dirichlet BC: boundary cells stay at their initial value. "
            "Row-major float32 storage of shape (NY, NX) — i indexes columns "
            "(fast axis), j indexes rows. Stable for alpha <= 0.25; we use "
            "alpha = 0.20 below the limit. The host runs the kernel for "
            "n_steps iterations with two buffers ping-ponged each call."
        ),
        kernel_signatures=(
            "kernel void heat_step(device const float *u_in  [[buffer(0)]],\n"
            "                      device       float *u_out [[buffer(1)]],\n"
            "                      constant uint      &NX    [[buffer(2)]],\n"
            "                      constant uint      &NY    [[buffer(3)]],\n"
            "                      constant float     &alpha [[buffer(4)]],\n"
            "                      uint2 gid [[thread_position_in_grid]]);\n"
            "\n"
            "Grid is dispatched 2-D as `threadsPerGrid = (NX, NY)`, one "
            "thread per output cell — guard with `if (i >= NX || j >= NY) "
            "return;`. Each thread MUST update exactly one cell; the host "
            "will not shrink the dispatch if you process multiple cells per "
            "thread, so extra threads just idle. Boundary cells (i==0, "
            "j==0, i==NX-1, j==NY-1) must copy u_in -> u_out unchanged."
        ),
        kernel_names=["heat_step"],
        seed_path=_SEED,
        sizes=[
            TaskSize("256x256_50",   {"nx": 256,  "ny": 256,  "n_steps": 50}),
            TaskSize("512x512_100",  {"nx": 512,  "ny": 512,  "n_steps": 100}),
            TaskSize("1024x1024_50", {"nx": 1024, "ny": 1024, "n_steps": 50}),
        ],
        held_out_sizes=[
            TaskSize("768x768_75", {"nx": 768, "ny": 768, "n_steps": 75}),
        ],
    )

    def evaluate_size(self, harness, pipelines, size, chip, n_warmup, n_measure):
        nx = int(size.params["nx"])
        ny = int(size.params["ny"])
        n_steps = int(size.params["n_steps"])
        alpha = 0.20

        u0 = _make_init(nx, ny)
        bA = harness.buf_from_np(u0)
        bB = harness.buf_zeros(u0.nbytes)
        bNX = harness.buf_scalar(nx, np.uint32)
        bNY = harness.buf_scalar(ny, np.uint32)
        bAlpha = harness.buf_scalar(alpha, np.float32)

        pso = pipelines["heat_step"]
        # 2D threadgroup: 16x16 is a robust default; check it fits.
        max_tg = int(pso.maxTotalThreadsPerThreadgroup())
        tg_w, tg_h = 16, 16
        while tg_w * tg_h > max_tg:
            tg_h //= 2
        grid_w = ((nx + tg_w - 1) // tg_w) * tg_w
        grid_h = ((ny + tg_h - 1) // tg_h) * tg_h

        view_A = harness.np_view(bA, np.float32, nx * ny)
        view_B = harness.np_view(bB, np.float32, nx * ny)

        def reset():
            view_A[:] = u0.ravel()
            view_B[:] = 0.0

        # We encode all n_steps dispatches into one command buffer so the
        # GPU timing covers the whole multi-step run.
        def dispatch(enc):
            enc.setComputePipelineState_(pso)
            # Buffers that don't change per step:
            enc.setBuffer_offset_atIndex_(bNX, 0, 2)
            enc.setBuffer_offset_atIndex_(bNY, 0, 3)
            enc.setBuffer_offset_atIndex_(bAlpha, 0, 4)
            for step in range(n_steps):
                if step % 2 == 0:
                    in_buf, out_buf = bA, bB
                else:
                    in_buf, out_buf = bB, bA
                enc.setBuffer_offset_atIndex_(in_buf, 0, 0)
                enc.setBuffer_offset_atIndex_(out_buf, 0, 1)
                enc.dispatchThreads_threadsPerThreadgroup_(
                    Metal.MTLSizeMake(grid_w, grid_h, 1),
                    Metal.MTLSizeMake(tg_w, tg_h, 1),
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

        # Final pass for correctness
        reset()
        harness.time_dispatch(dispatch)
        final_buf_view = view_A if n_steps % 2 == 0 else view_B
        got = final_buf_view.copy().reshape(ny, nx)

        expected = _cpu_reference(u0, alpha, n_steps)
        err = float(np.max(np.abs(got - expected)))
        # Tolerance: 1e-4 absolute is generous for 100-step fp32 stencils.
        tol = 1e-4 + 1e-5 * float(np.max(np.abs(expected)))
        correct = err <= tol

        # BW-bound roofline: each interior cell loads 5 floats + stores 1
        # = 24 bytes, but with perfect cache reuse the *unique* DRAM
        # traffic is 4 bytes load + 4 bytes store per cell per step. We
        # use the optimistic 8 bytes/cell number for the ceiling so that
        # only the very best implementations hit ~100%.
        bytes_per_step = 8.0 * nx * ny
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
            achieved_unit="GB/s (effective, 8 B/cell)",
            ceiling=ceiling,
            ceiling_unit="GB/s",
            fraction_of_ceiling=achieved / ceiling if ceiling > 0 else 0.0,
            extra={"tol": tol, "nx": nx, "ny": ny, "n_steps": n_steps,
                   "alpha": alpha},
        )
