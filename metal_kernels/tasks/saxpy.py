"""SAXPY: y = a*x + y.

Memory-bound; serves as a smoke test for the harness. Each element does
one fused multiply-add (~2 FLOPs) but reads/writes 12 bytes (load x, load
y, store y) — arithmetic intensity 1/6, deeply BW-bound.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import Metal

from ..harness import MetalHarness, grid_1d, threadgroup_1d
from ..hardware import ChipSpec
from ..task import (
    Task, TaskSize, TaskSpec, SizeResult,
    gb_per_s, register_task,
)


_SEED = Path(__file__).resolve().parent.parent.parent / "seeds" / "saxpy.metal"


@register_task("saxpy")
class SaxpyTask(Task):
    spec = TaskSpec(
        name="saxpy",
        description=(
            "SAXPY: out-of-place y = a*x + y. Memory-bound; expected to be "
            "BW-bound on Apple Silicon. Bytes moved per element = 12 "
            "(load x, load y, store y)."
        ),
        kernel_signatures=(
            "kernel void saxpy(device const float *x [[buffer(0)]],\n"
            "                  device float       *y [[buffer(1)]],\n"
            "                  constant float     &a [[buffer(2)]],\n"
            "                  constant uint      &N [[buffer(3)]],\n"
            "                  uint i [[thread_position_in_grid]]);\n"
            "\n"
            "Update y[i] = a * x[i] + y[i] for i in [0, N). Threads are "
            "dispatched 1-D, one per element (grid is padded up to a "
            "multiple of the threadgroup width, so guard against i >= N). "
            "Each thread MUST handle exactly one i; the host will not "
            "shrink the dispatch if you process multiple elements per "
            "thread — extra threads just idle."
        ),
        kernel_names=["saxpy"],
        seed_path=_SEED,
        sizes=[
            TaskSize("1M",  {"n": 1 << 20}),
            TaskSize("16M", {"n": 1 << 24}),
            TaskSize("64M", {"n": 1 << 26}),
        ],
        held_out_sizes=[
            TaskSize("4M",  {"n": 1 << 22}),
        ],
    )

    def evaluate_size(self, harness, pipelines, size, chip, n_warmup, n_measure):
        n = int(size.params["n"])
        rng = np.random.default_rng(0xBEEF)
        x = rng.standard_normal(n).astype(np.float32)
        y0 = rng.standard_normal(n).astype(np.float32)
        a = np.float32(2.5)

        bx = harness.buf_from_np(x)
        by = harness.buf_from_np(y0)
        ba = harness.buf_scalar(a, np.float32)
        bn = harness.buf_scalar(n, np.uint32)

        pso = pipelines["saxpy"]
        # Choose threadgroup size: cap at threadExecutionWidth*8, max 1024.
        tew = int(pso.threadExecutionWidth())
        max_tg = int(pso.maxTotalThreadsPerThreadgroup())
        tg_w = min(max_tg, max(tew, 256))
        # Round grid up to a multiple of tg_w so the LLM-written kernel can
        # use thread_position_in_grid + an `if i >= N` guard.
        grid_w = ((n + tg_w - 1) // tg_w) * tg_w

        def dispatch(enc):
            enc.setComputePipelineState_(pso)
            enc.setBuffer_offset_atIndex_(bx, 0, 0)
            enc.setBuffer_offset_atIndex_(by, 0, 1)
            enc.setBuffer_offset_atIndex_(ba, 0, 2)
            enc.setBuffer_offset_atIndex_(bn, 0, 3)
            enc.dispatchThreads_threadsPerThreadgroup_(
                Metal.MTLSizeMake(grid_w, 1, 1),
                Metal.MTLSizeMake(tg_w, 1, 1),
            )

        # Time first (with warmup); then verify on a fresh y to avoid the
        # warmup runs accumulating into y. We re-init y between runs by
        # uploading a fresh copy each iteration via a wrapping closure.
        def timed_dispatch(enc):
            # Restore y before each timed dispatch so the result is
            # deterministic. This adds a small CPU→GPU memcpy to each
            # iteration; for BW-bound kernels at these sizes it's
            # negligible compared to the dispatch itself, but to be safe
            # we'll restore *outside* the encoder by reuploading the
            # buffer between dispatches (we time a single dispatch).
            enc.setComputePipelineState_(pso)
            enc.setBuffer_offset_atIndex_(bx, 0, 0)
            enc.setBuffer_offset_atIndex_(by, 0, 1)
            enc.setBuffer_offset_atIndex_(ba, 0, 2)
            enc.setBuffer_offset_atIndex_(bn, 0, 3)
            enc.dispatchThreads_threadsPerThreadgroup_(
                Metal.MTLSizeMake(grid_w, 1, 1),
                Metal.MTLSizeMake(tg_w, 1, 1),
            )

        # Manual timing loop: between each dispatch, restore y in unified
        # memory (costs ~one memcpy of size 4N bytes; cheap relative to GPU
        # time at large N because LLDB et al run on perf cores).
        view_y = harness.np_view(by, np.float32, n)

        # Warmup
        for _ in range(n_warmup):
            view_y[:] = y0
            harness.time_dispatch(dispatch)
        # Measured
        samples = []
        for _ in range(n_measure):
            view_y[:] = y0
            samples.append(harness.time_dispatch(dispatch))
        gpu_s = float(np.median(samples))

        # Verify: re-run once to capture the final state
        view_y[:] = y0
        harness.time_dispatch(dispatch)
        got = view_y.copy()
        expected = a * x + y0
        err = float(np.max(np.abs(got - expected)))
        # SAXPY in fp32: error should be O(eps * max|expected|).
        tol = 1e-5 * max(1.0, float(np.max(np.abs(expected))))
        correct = err <= tol

        bytes_moved = 12.0 * n  # load x + load y + store y
        achieved = gb_per_s(bytes_moved, gpu_s)
        ceiling = float(chip.peak_bw_gb_s)
        return SizeResult(
            size_label=size.label,
            correct=correct,
            error_value=err,
            error_kind="max_abs",
            gpu_seconds=gpu_s,
            achieved=achieved,
            achieved_unit="GB/s",
            ceiling=ceiling,
            ceiling_unit="GB/s",
            fraction_of_ceiling=achieved / ceiling if ceiling > 0 else 0.0,
            extra={"tol": tol, "n": n},
        )
