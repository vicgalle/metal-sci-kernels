"""Lattice Boltzmann D2Q9 (Plan §3, Task 5).

Combined pull-stream + BGK collision in a single kernel. Periodic BCs are
used as the deterministic correctness gate; physical lid-driven-cavity
validation against Ghia 1982 is deferred.

The candidate kernel must respect the SoA storage layout for buffers 0
and 1: ``f[k * NX*NY + j*NX + i]``. Internal experimentation with AoS or
AA-pattern is fine as long as the buffers exposed to the host follow
this convention. The host ping-pongs two buffers across ``n_steps``,
encoding all dispatches into a single command buffer for accurate
timing.
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


_SEED = Path(__file__).resolve().parent.parent.parent / "seeds" / "lbm.metal"


# D2Q9 directions and weights — kept identical to the kernel's `constant` arrays.
_CX = np.array([0,  1,  0, -1,  0,  1, -1, -1,  1], dtype=np.int32)
_CY = np.array([0,  0,  1,  0, -1,  1,  1, -1, -1], dtype=np.int32)
_W  = np.array([4/9,
                1/9, 1/9, 1/9, 1/9,
                1/36, 1/36, 1/36, 1/36], dtype=np.float32)


def _make_init(nx: int, ny: int) -> np.ndarray:
    """SoA f field of shape (9, NY, NX), float32.

    Initial condition: small Gaussian density bump at the centre, zero
    velocity. f_k = w_k * rho. Smooth, well-conditioned, exercises every
    streaming direction once the bump diffuses.
    """
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    cx_ = (nx - 1) / 2.0
    cy_ = (ny - 1) / 2.0
    sigma = 0.15 * min(nx, ny)
    bump = np.exp(-(((xx - cx_) ** 2 + (yy - cy_) ** 2) / (2 * sigma ** 2)))
    rho = (1.0 + 0.05 * bump).astype(np.float32)
    f = np.empty((9, ny, nx), dtype=np.float32)
    for k in range(9):
        f[k] = _W[k] * rho
    return np.ascontiguousarray(f)


def _cpu_reference(f0: np.ndarray, tau: float, n_steps: int) -> np.ndarray:
    """Vectorized numpy reference. Same operation order as the GPU kernel.

    f0: (9, NY, NX) float32. Returns f after n_steps of pull-stream + BGK.
    """
    f = f0.astype(np.float32, copy=True)
    inv_tau = np.float32(1.0 / tau)
    for _ in range(n_steps):
        # Pull streaming with periodic wrap. np.roll(arr, shift=cy, axis=0)
        # is equivalent to "result[y, x] = arr[(y - cy) mod N, x]" — i.e.
        # the cell at y receives the value from y - cy, exactly the pull.
        streamed = np.empty_like(f)
        for k in range(9):
            streamed[k] = np.roll(
                f[k], shift=(int(_CY[k]), int(_CX[k])), axis=(0, 1),
            )

        # Moments — match GPU's sequential reduction order so that fp32
        # rounding is identical (or as close as practical).
        rho = np.zeros_like(streamed[0])
        ux = np.zeros_like(streamed[0])
        uy = np.zeros_like(streamed[0])
        for k in range(9):
            rho = rho + streamed[k]
            ux  = ux  + np.float32(_CX[k]) * streamed[k]
            uy  = uy  + np.float32(_CY[k]) * streamed[k]
        ux = ux / rho
        uy = uy / rho

        usq = ux * ux + uy * uy
        out = np.empty_like(f)
        for k in range(9):
            cu = np.float32(_CX[k]) * ux + np.float32(_CY[k]) * uy
            feq = _W[k] * rho * (
                np.float32(1.0)
                + np.float32(3.0) * cu
                + np.float32(4.5) * cu * cu
                - np.float32(1.5) * usq
            )
            out[k] = streamed[k] - inv_tau * (streamed[k] - feq)
        f = out
    return f


@register_task("lbm")
class LBMD2Q9Task(Task):
    spec = TaskSpec(
        name="lbm",
        description=(
            "D2Q9 lattice Boltzmann method, fused pull-streaming + BGK "
            "collision, periodic boundary conditions. Distribution functions "
            "are stored SoA: f[k * NX*NY + j*NX + i] for k in [0, 9), "
            "j in [0, NY), i in [0, NX), float32 row-major.\n\n"
            "Per timestep, per cell (i, j):\n"
            "  1) PULL stream: f_streamed[k] = f_in[k, (i - CX[k]) mod NX,\n"
            "                                          (j - CY[k]) mod NY]\n"
            "  2) Moments: rho = sum_k f_streamed[k];\n"
            "     u = (sum_k CX[k] * f_streamed[k]) / rho; v likewise.\n"
            "  3) BGK collision: f_out[k] = f_streamed[k]\n"
            "       - (1/tau) (f_streamed[k] - f_eq[k])\n"
            "     with f_eq[k] = W[k] * rho *\n"
            "       (1 + 3 (CX[k] u + CY[k] v)\n"
            "          + 4.5 (CX[k] u + CY[k] v)^2 - 1.5 (u^2 + v^2)).\n"
            "Velocity table CX[9] = {0, 1, 0,-1, 0, 1,-1,-1, 1};\n"
            "                CY[9] = {0, 0, 1, 0,-1, 1, 1,-1,-1};\n"
            "weights W[9] = {4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36}.\n\n"
            "The host runs the kernel n_steps times with two buffers "
            "ping-ponged each call. Effective DRAM traffic per step is "
            "72 bytes/cell (9 reads + 9 writes), so the roofline is BW-bound."
        ),
        kernel_signatures=(
            "kernel void lbm_step(device const float *f_in   [[buffer(0)]],\n"
            "                     device       float *f_out  [[buffer(1)]],\n"
            "                     constant uint        &NX   [[buffer(2)]],\n"
            "                     constant uint        &NY   [[buffer(3)]],\n"
            "                     constant float       &tau  [[buffer(4)]],\n"
            "                     uint2 gid [[thread_position_in_grid]]);\n"
            "\n"
            "Grid is dispatched 2-D as `threadsPerGrid = (NX, NY)`, one "
            "thread per output cell — guard with `if (i >= NX || j >= NY) "
            "return;`. Each thread MUST update exactly one cell; the host "
            "will not shrink the dispatch if you process multiple cells "
            "per thread, so extra threads just idle. SoA layout MUST be "
            "preserved on buffers 0 and 1; the kernel may use any internal "
            "layout/optimization (threadgroup tiling, simdgroup ops, etc.)."
        ),
        kernel_names=["lbm_step"],
        seed_path=_SEED,
        sizes=[
            TaskSize("64x64_50",    {"nx": 64,  "ny": 64,  "n_steps": 50}),
            TaskSize("128x128_100", {"nx": 128, "ny": 128, "n_steps": 100}),
            TaskSize("256x256_100", {"nx": 256, "ny": 256, "n_steps": 100}),
        ],
        held_out_sizes=[
            TaskSize("192x192_75",  {"nx": 192, "ny": 192, "n_steps": 75}),
        ],
    )

    def evaluate_size(self, harness, pipelines, size, chip, n_warmup, n_measure):
        nx = int(size.params["nx"])
        ny = int(size.params["ny"])
        n_steps = int(size.params["n_steps"])
        tau = 0.8

        f0 = _make_init(nx, ny)            # (9, NY, NX) fp32
        nbytes = f0.nbytes

        bA = harness.buf_from_np(f0)
        bB = harness.buf_zeros(nbytes)
        bNX = harness.buf_scalar(nx, np.uint32)
        bNY = harness.buf_scalar(ny, np.uint32)
        bTau = harness.buf_scalar(tau, np.float32)

        pso = pipelines["lbm_step"]
        max_tg = int(pso.maxTotalThreadsPerThreadgroup())
        # 16x16 is a robust default; 256 threads per group fits everywhere.
        tg_w, tg_h = 16, 16
        while tg_w * tg_h > max_tg:
            tg_h //= 2
        grid_w = ((nx + tg_w - 1) // tg_w) * tg_w
        grid_h = ((ny + tg_h - 1) // tg_h) * tg_h

        view_A = harness.np_view(bA, np.float32, 9 * nx * ny)
        view_B = harness.np_view(bB, np.float32, 9 * nx * ny)

        def reset():
            view_A[:] = f0.ravel()
            view_B[:] = 0.0

        def dispatch(enc):
            enc.setComputePipelineState_(pso)
            enc.setBuffer_offset_atIndex_(bNX, 0, 2)
            enc.setBuffer_offset_atIndex_(bNY, 0, 3)
            enc.setBuffer_offset_atIndex_(bTau, 0, 4)
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

        # Final correctness pass
        reset()
        harness.time_dispatch(dispatch)
        final_view = view_A if n_steps % 2 == 0 else view_B
        got = final_view.copy().reshape(9, ny, nx)

        expected = _cpu_reference(f0, tau, n_steps)
        max_ref = float(np.max(np.abs(expected)))
        err = float(np.max(np.abs(got - expected)))
        # Mixed abs+rel tolerance. fp32 rounding accumulates over n_steps;
        # 1e-4 absolute is comfortably above the observed CPU/GPU drift
        # without masking real bugs.
        tol = 5e-4 + 1e-4 * max_ref
        correct = err <= tol

        # Roofline: 9 reads + 9 writes per cell per step = 72 B/cell.
        bytes_per_step = 72.0 * nx * ny
        bytes_total = bytes_per_step * n_steps
        achieved = gb_per_s(bytes_total, gpu_s)
        ceiling = float(chip.peak_bw_gb_s)
        return SizeResult(
            size_label=size.label,
            correct=correct,
            error_value=err,
            error_kind="max_abs_f",
            gpu_seconds=gpu_s,
            achieved=achieved,
            achieved_unit="GB/s (effective, 72 B/cell)",
            ceiling=ceiling,
            ceiling_unit="GB/s",
            fraction_of_ceiling=achieved / ceiling if ceiling > 0 else 0.0,
            extra={"tol": tol, "nx": nx, "ny": ny,
                   "n_steps": n_steps, "tau": tau},
        )
