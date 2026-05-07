"""3D complex-to-complex forward FFT, fp32, power-of-two cubes.

Three kernels (one per axis); the host calls them in sequence
``fft3d_x → fft3d_y → fft3d_z`` ping-ponging between two ``float2`` buffers.
Each kernel does one length-N 1D FFT per threadgroup using shared memory:
cooperative load with bit-reversal permutation, ``log2(N)`` Cooley-Tukey
butterfly stages.

Optimization regime: this is the suite's first task whose dominant lever is
*data movement and shuffle* rather than stencil tiling, register blocking,
or atomic scatter. The optimization surface includes:

  * Stockham auto-sort (no bit-reversal pass) vs Cooley-Tukey + bit-reverse
  * twiddle factor caching (precomputed table) vs sin/cos on the fly
  * mixed-radix (radix-4, radix-8) butterflies for fewer barriers
  * `simd_shuffle` / `simd_shuffle_xor` for intra-simdgroup butterflies
    instead of threadgroup memory and a barrier
  * bank-conflict avoidance via padded threadgroup-memory strides
  * per-axis kernel fusion (e.g. fold the z-pass with a transpose)

Roofline: BW-bound at three full passes through the volume (16 B/cell read
+ 16 B/cell write per pass, 48 B/cell total). At our smallest size 32^3
the full working set is ~256 KB and SLC-resident, so the achieved/ceiling
fraction will exceed 100% — same flag as the LBM 256^2 caveat. The
mid/large sizes (64^3, 128^3) DRAM-bind cleanly.
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


_SEED = Path(__file__).resolve().parent.parent.parent / "seeds" / "fft3d.metal"


def _make_input(N: int, seed: int = 0xF17F) -> np.ndarray:
    """Random complex Gaussian, shape (N, N, N), fp32 interleaved (re, im).

    Returns a float32 array of shape (N, N, N, 2) — the (..., 0) channel
    is real and (..., 1) is imag, matching Metal's ``float2`` layout.
    """
    rng = np.random.default_rng(seed)
    re = rng.standard_normal(size=(N, N, N), dtype=np.float32)
    im = rng.standard_normal(size=(N, N, N), dtype=np.float32)
    out = np.stack([re, im], axis=-1).astype(np.float32)
    return np.ascontiguousarray(out)


def _cpu_reference(x: np.ndarray) -> np.ndarray:
    """numpy 3D forward FFT, unnormalized. Input/output (N, N, N, 2) fp32."""
    z = x[..., 0] + 1j * x[..., 1]
    Y = np.fft.fftn(z, axes=(0, 1, 2)).astype(np.complex64)
    out = np.empty_like(x)
    out[..., 0] = Y.real
    out[..., 1] = Y.imag
    return out


@register_task("fft3d")
class FFT3DTask(Task):
    spec = TaskSpec(
        name="fft3d",
        description=(
            "3D complex-to-complex forward FFT, fp32, on a power-of-two cube "
            "of side N. Convention: forward, unnormalized — \n"
            "  Y[k,j,i] = sum_{kk,jj,ii} X[kk,jj,ii] * "
            "exp(-2πi (k·kk + j·jj + i·ii) / N)\n"
            "(matches numpy.fft.fftn with norm='backward').\n\n"
            "Storage is row-major float2[NZ][NY][NX] with NX=NY=NZ=N. "
            "Linear index of element (i,j,k) is ((k·N + j)·N + i); float2 "
            "is (real, imag) and is the buffer element type. The host "
            "calls three separate kernels — fft3d_x, fft3d_y, fft3d_z — "
            "in that order, ping-ponging between two device buffers (so "
            "the final 3D FFT result lands in the second buffer). Each "
            "kernel does one 1D length-N FFT per threadgroup; the FFT "
            "axis is fixed by the kernel name and its index decoding.\n\n"
            "Because the three axes are orthogonal, the FFTs commute — "
            "the result is invariant to the order x→y→z vs any other "
            "order, but the host fixes the order x→y→z and the kernel "
            "names must match. The optimization surface is dominated by "
            "data movement: bit-reversal vs Stockham auto-sort, twiddle "
            "caching, simdgroup-shuffle butterflies, and threadgroup-"
            "memory bank-conflict avoidance are all on the table."
        ),
        kernel_signatures=(
            "kernel void fft3d_x(device const float2 *in_data  [[buffer(0)]],\n"
            "                    device       float2 *out_data [[buffer(1)]],\n"
            "                    constant uint        &N        [[buffer(2)]],\n"
            "                    uint3 gid [[thread_position_in_grid]]);\n"
            "kernel void fft3d_y(device const float2 *in_data  [[buffer(0)]],\n"
            "                    device       float2 *out_data [[buffer(1)]],\n"
            "                    constant uint        &N        [[buffer(2)]],\n"
            "                    uint3 gid [[thread_position_in_grid]]);\n"
            "kernel void fft3d_z(device const float2 *in_data  [[buffer(0)]],\n"
            "                    device       float2 *out_data [[buffer(1)]],\n"
            "                    constant uint        &N        [[buffer(2)]],\n"
            "                    uint3 gid [[thread_position_in_grid]]);\n"
            "\n"
            "Dispatch geometry (identical for all three kernels, host-fixed):\n"
            "  threadsPerGrid        = (N, N*N, 1)\n"
            "  threadsPerThreadgroup = (N, 1,   1)\n"
            "So each TG of N threads owns exactly one length-N line: gid.x "
            "is the position along the FFT axis (= thread_position_in_"
            "threadgroup.x) and gid.y indexes the (N×N) plane of lines "
            "orthogonal to that axis. Index decoding per kernel:\n"
            "  fft3d_x: i = gid.x; k = gid.y / N; j = gid.y - k*N\n"
            "  fft3d_y: j = gid.x; k = gid.y / N; i = gid.y - k*N\n"
            "  fft3d_z: k = gid.x; j = gid.y / N; i = gid.y - j*N\n"
            "Each TG must produce the full FFT of its line in out_data; "
            "the host runs the three kernels back-to-back in one command "
            "buffer and ping-pongs the buffers, so out_data of one pass "
            "is the in_data of the next.\n\n"
            "If you cap the threadgroup with [[max_total_threads_per_"
            "threadgroup(N)]], place the attribute on the kernel "
            "declaration line itself (not as a free-standing statement), "
            "and remember the host's TG width along x is N (≤ 1024 on "
            "M-series). Your tile / shared-memory layout MUST match the "
            "dispatched (N, 1, 1) TG geometry: the host will not split "
            "or reshape the dispatch to fit a different tile."
        ),
        kernel_names=["fft3d_x", "fft3d_y", "fft3d_z"],
        seed_path=_SEED,
        sizes=[
            # 32^3 = 256 KB working set (SLC-resident, compute-bound regime);
            # 64^3 = 2 MB (still SLC); 128^3 = 16 MB (DRAM-bound).
            TaskSize("32cube",  {"n": 32}),
            TaskSize("64cube",  {"n": 64}),
            TaskSize("128cube", {"n": 128}),
        ],
        held_out_sizes=[
            TaskSize("256cube", {"n": 256}),  # held out: extrapolation past the largest training size
        ],
    )

    def evaluate_size(self, harness, pipelines, size, chip, n_warmup, n_measure):
        N = int(size.params["n"])
        # The seed (and any plain radix-2 candidate) requires N to be a
        # power of two and N <= 1024 (max threadgroup width on M-series).
        if (N & (N - 1)) != 0:
            raise ValueError(f"fft3d size n={N} must be a power of two")
        if N > 1024:
            raise ValueError(f"fft3d size n={N} exceeds 1024 (max TG width)")

        x0 = _make_input(N)                     # (N, N, N, 2) fp32
        n_complex = N * N * N
        nbytes = x0.nbytes                      # = n_complex * 8

        bA = harness.buf_from_np(x0)
        bB = harness.buf_zeros(nbytes)
        bN = harness.buf_scalar(N, np.uint32)

        pso_x = pipelines["fft3d_x"]
        pso_y = pipelines["fft3d_y"]
        pso_z = pipelines["fft3d_z"]

        # Each TG owns one length-N line, with N threads. If a candidate
        # ships a [[max_total_threads_per_threadgroup]] cap below N, we
        # honour it — but since the algorithm needs N cooperating threads
        # per line, that's almost certainly a bug; we fail loudly later
        # via a correctness check rather than silently shrinking.
        for pso, name in [(pso_x, "fft3d_x"), (pso_y, "fft3d_y"), (pso_z, "fft3d_z")]:
            if int(pso.maxTotalThreadsPerThreadgroup()) < N:
                raise RuntimeError(
                    f"{name} declares max_total_threads_per_threadgroup="
                    f"{int(pso.maxTotalThreadsPerThreadgroup())} which is "
                    f"smaller than the host-dispatched TG width N={N}; the "
                    f"FFT line cannot fit in one threadgroup"
                )

        grid = Metal.MTLSizeMake(N, N * N, 1)
        tg = Metal.MTLSizeMake(N, 1, 1)

        view_A = harness.np_view(bA, np.float32, n_complex * 2)
        view_B = harness.np_view(bB, np.float32, n_complex * 2)

        def reset():
            view_A[:] = x0.ravel()
            view_B[:] = 0.0

        # Three passes, ping-pong A→B (x), B→A (y), A→B (z). Final lands in B.
        passes = [(pso_x, bA, bB), (pso_y, bB, bA), (pso_z, bA, bB)]

        def dispatch(enc):
            for pso, src, dst in passes:
                enc.setComputePipelineState_(pso)
                enc.setBuffer_offset_atIndex_(src, 0, 0)
                enc.setBuffer_offset_atIndex_(dst, 0, 1)
                enc.setBuffer_offset_atIndex_(bN, 0, 2)
                enc.dispatchThreads_threadsPerThreadgroup_(grid, tg)

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

        # Final correctness pass.
        reset()
        harness.time_dispatch(dispatch)
        got = view_B.copy().reshape(N, N, N, 2)

        expected = _cpu_reference(x0)
        max_ref = float(np.max(np.abs(expected)))
        err = float(np.max(np.abs(got - expected)))
        # fp32 FFT accumulates ~O(eps · sqrt(N) · log2(N)) per output; with
        # input variance 1, output magnitudes scale as sqrt(N^3). At N=128,
        # eps · sqrt(N^3) · log2(N) ≈ 1e-3 — relative tolerance dominates.
        tol = 1e-3 + 1e-3 * max_ref
        correct = err <= tol

        # BW-bound roofline. Three axis passes; each pass reads N^3
        # complex (16 B/cell) and writes N^3 complex (16 B/cell). Effective
        # DRAM traffic = 96 B per cell summed across 3 passes ... but the
        # host ping-pongs between two physical buffers, so each pass really
        # is a full DRAM read + write of the volume. 48 B/cell is the
        # cache-perfect three-pass total (16 read + 16 write × 3 / 3? no).
        # Concretely: pass k touches 16 B read + 16 B write per cell, so
        # the *whole 3D FFT* moves 3 · 32 = 96 B per cell at the streamed
        # limit. We report against this.
        bytes_total = 96.0 * n_complex
        achieved = gb_per_s(bytes_total, gpu_s)
        ceiling = float(chip.peak_bw_gb_s)

        # Also report achieved GFLOPS (5 N log2 N per 1D FFT, N^2 lines per
        # axis × 3 axes = 15 · N^3 · log2(N)) for the extras dict.
        log2N = int(np.log2(N))
        flops = 15.0 * n_complex * log2N
        achieved_gflops = flops / gpu_s / 1e9

        return SizeResult(
            size_label=size.label,
            correct=correct,
            error_value=err,
            error_kind="max_abs",
            gpu_seconds=gpu_s,
            achieved=achieved,
            achieved_unit="GB/s (effective, 96 B/cell across 3 passes)",
            ceiling=ceiling,
            ceiling_unit="GB/s",
            fraction_of_ceiling=achieved / ceiling if ceiling > 0 else 0.0,
            extra={
                "tol": tol, "n": N, "max_ref": max_ref,
                "achieved_gflops": achieved_gflops,
                "flops": flops,
            },
        )
