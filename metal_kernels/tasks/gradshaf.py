"""Grad-Shafranov fixed-boundary Picard-Jacobi iteration.

Two kernels per outer step (a multi-kernel task — first one in the suite):

1. ``gradshaf_axis_reduce``: max-reduction over the interior of psi into a
   single-scalar buffer ``psi_axis``.
2. ``gradshaf_step``: per-cell stencil + nonlinear source. With Dirichlet
   psi=0 on the boundary, ψ_norm = ψ/ψ_axis. Source
   ``J(R, ψ_norm) = R · p_axis · 4 ψ_norm (1 − ψ_norm)`` masked outside
   (0, 1). Δ* uses R-dependent east/west weights and constant N/S/C
   weights. Jacobi update with under-relaxation ω.

The host runs K Picard outer steps, alternating reduce → step in a single
command buffer for accurate end-to-end GPU timing. Buffers ping-pong:
psi_in / psi_out swap each outer step; psi_axis is reused.

Optimization regime: this task is the first in the suite to combine an
in-kernel reduction with a stencil. It also introduces a variable-
coefficient stencil (the 1/R term breaks east/west symmetry) and
multi-kernel composition with implicit dependency ordering.
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


_SEED = Path(__file__).resolve().parent.parent.parent / "seeds" / "gradshaf.metal"


# Domain and physical constants are baked into the task — the LLM doesn't
# tune these. Dimensionless: μ₀ and p_axis are chosen so the source
# materially shapes ψ (visible Shafranov-shift-like asymmetry vs. p_axis=1)
# without driving the Picard iteration unstable. ω = 1.0 is plain Jacobi.
_RMIN = 1.0
_RMAX = 2.0
_ZMIN = -0.5
_ZMAX = 0.5
_P_AXIS = 200.0
_MU0 = 1.0
_OMEGA = 1.0
_REDUCE_TG = 256  # naive single-TG reduction width


def _make_init(NR: int, NZ: int) -> np.ndarray:
    """Sin-bump initial ψ: smooth, peak ~1 at centre, zero on boundary."""
    u = np.linspace(0.0, 1.0, NR, dtype=np.float32)
    v = np.linspace(0.0, 1.0, NZ, dtype=np.float32)
    psi = (np.sin(np.pi * v)[:, None] * np.sin(np.pi * u)[None, :]).astype(np.float32)
    # Clamp the boundary exactly to zero so Dirichlet is bit-exact preserved
    # by the kernel's "copy through" branch.
    psi[0, :] = 0.0
    psi[-1, :] = 0.0
    psi[:, 0] = 0.0
    psi[:, -1] = 0.0
    return np.ascontiguousarray(psi)


def _cpu_reference(psi0: np.ndarray, NR: int, NZ: int, K: int,
                   *, Rmin: float, Rmax: float, Zmin: float, Zmax: float,
                   p_axis: float, mu0: float, omega: float) -> np.ndarray:
    """Numpy reference. Mirrors the kernel's per-cell op order so fp32
    rounding is identical (or as close as practical) across CPU and GPU.
    """
    dR = np.float32((Rmax - Rmin) / (NR - 1))
    dZ = np.float32((Zmax - Zmin) / (NZ - 1))
    R = (Rmin + np.arange(NR, dtype=np.float32) * dR).astype(np.float32)

    inv_dR2 = np.float32(1.0) / (dR * dR)
    inv_dZ2 = np.float32(1.0) / (dZ * dZ)
    h_inv_RdR = (np.float32(0.5) / (R * dR)).astype(np.float32)
    a_W = (inv_dR2 + h_inv_RdR).astype(np.float32)        # (NR,)
    a_E = (inv_dR2 - h_inv_RdR).astype(np.float32)        # (NR,)
    a_N = inv_dZ2
    a_S = inv_dZ2
    a_C = (np.float32(-2.0) * inv_dR2 - np.float32(2.0) * inv_dZ2).astype(np.float32)

    Rint = R[1:-1][None, :]                                # (1, NR-2)
    a_W_int = a_W[1:-1][None, :]                           # (1, NR-2)
    a_E_int = a_E[1:-1][None, :]                           # (1, NR-2)
    p_axis_f = np.float32(p_axis)
    mu0_f = np.float32(mu0)
    omega_f = np.float32(omega)

    psi = psi0.astype(np.float32, copy=True)
    psi_new = psi.copy()
    for _ in range(K):
        psi_axis = np.float32(psi[1:-1, 1:-1].max())
        psi_C = psi[1:-1, 1:-1]
        psi_norm = (psi_C / psi_axis).astype(np.float32)
        in_plasma = (psi_norm > np.float32(0.0)) & (psi_norm < np.float32(1.0))
        J = np.where(
            in_plasma,
            (Rint * p_axis_f * np.float32(4.0)
             * psi_norm * (np.float32(1.0) - psi_norm)),
            np.float32(0.0),
        ).astype(np.float32)
        rhs = (-mu0_f * Rint * J).astype(np.float32)

        # Same operation order as the kernel: a_W·ψ_W + a_E·ψ_E + a_N·ψ_N
        # + a_S·ψ_S + a_C·ψ_C, all left-to-right additions.
        delta = (a_W_int * psi[1:-1, :-2]).astype(np.float32)
        delta = (delta + a_E_int * psi[1:-1, 2:]).astype(np.float32)
        delta = (delta + a_N * psi[:-2, 1:-1]).astype(np.float32)
        delta = (delta + a_S * psi[2:, 1:-1]).astype(np.float32)
        delta = (delta + a_C * psi_C).astype(np.float32)
        r = (rhs - delta).astype(np.float32)
        psi_new[1:-1, 1:-1] = (psi_C + omega_f * r / a_C).astype(np.float32)
        # Boundary copy: psi0 already has 0 on the boundary, and the kernel
        # copies through unchanged, so psi_new boundary stays zero.
        psi_new[0, :] = psi[0, :]
        psi_new[-1, :] = psi[-1, :]
        psi_new[:, 0] = psi[:, 0]
        psi_new[:, -1] = psi[:, -1]
        psi, psi_new = psi_new, psi
    return psi


@register_task("gradshaf")
class GradShafranovTask(Task):
    spec = TaskSpec(
        name="gradshaf",
        description=(
            "Grad-Shafranov fixed-boundary equilibrium via K Picard outer "
            "steps. Per outer step:\n\n"
            "  1. ψ_axis = max over INTERIOR of ψ      (i in [1, NR-1), "
            "                                           j in [1, NZ-1))\n"
            "  2. For each interior (i, j):\n"
            "       R         = Rmin + i*dR\n"
            "       ψ_norm    = ψ[j,i] / ψ_axis\n"
            "       J         = (0 < ψ_norm < 1) ? R * p_axis * 4 ψ_norm (1 − ψ_norm) : 0\n"
            "       rhs       = −μ₀ * R * J\n"
            "       Δ*ψ       = a_W ψ[j,i-1] + a_E ψ[j,i+1]\n"
            "                 + a_N ψ[j+1,i] + a_S ψ[j-1,i] + a_C ψ[j,i]\n"
            "         a_W = 1/dR² + 1/(2 R dR)     (R-dependent: 1/R term)\n"
            "         a_E = 1/dR² − 1/(2 R dR)\n"
            "         a_N = a_S = 1/dZ²\n"
            "         a_C = −2/dR² − 2/dZ²\n"
            "       r         = rhs − Δ*ψ\n"
            "       ψ_new[j,i] = ψ[j,i] + ω * r / a_C\n"
            "  3. Boundary cells (i==0, j==0, i==NR-1, j==NZ-1) MUST copy "
            "     ψ_in -> ψ_out unchanged (Dirichlet ψ=0 is preserved).\n\n"
            "Storage is row-major float32 of shape (NZ, NR): linear index "
            "= j*NR + i, with i the fast (R) axis. Domain is fixed at "
            f"R ∈ [{_RMIN}, {_RMAX}], Z ∈ [{_ZMIN}, {_ZMAX}]; "
            f"μ₀={_MU0}, p_axis={_P_AXIS}, ω={_OMEGA} are dimensionless "
            "and shared across all sizes. The host calls "
            "gradshaf_axis_reduce → gradshaf_step in alternation for K "
            "outer steps within one command buffer; psi_in/psi_out "
            "ping-pong each step. The reduction's output buffer "
            "(psi_axis) is a single-scalar device buffer that the host "
            "rebinds for each outer step. Effective DRAM traffic per "
            "outer step is ~12 B/cell (4 B reduction read + 8 B stencil "
            "read+write); the roofline is BW-bound."
        ),
        kernel_signatures=(
            "kernel void gradshaf_axis_reduce("
            "device const float *psi      [[buffer(0)]],\n"
            "                                 device       float *psi_axis "
            "[[buffer(1)]],\n"
            "                                 constant uint       &NR      "
            "[[buffer(2)]],\n"
            "                                 constant uint       &NZ      "
            "[[buffer(3)]],\n"
            "                                 uint tid                     "
            "[[thread_position_in_threadgroup]],\n"
            "                                 uint tgsize                  "
            "[[threads_per_threadgroup]]);\n"
            "\n"
            "kernel void gradshaf_step("
            "device const float *psi_in   [[buffer(0)]],\n"
            "                          device       float *psi_out  "
            "[[buffer(1)]],\n"
            "                          device const float *psi_axis "
            "[[buffer(2)]],\n"
            "                          constant uint       &NR      "
            "[[buffer(3)]],\n"
            "                          constant uint       &NZ      "
            "[[buffer(4)]],\n"
            "                          constant float      &Rmin    "
            "[[buffer(5)]],\n"
            "                          constant float      &dR      "
            "[[buffer(6)]],\n"
            "                          constant float      &dZ      "
            "[[buffer(7)]],\n"
            "                          constant float      &p_axis  "
            "[[buffer(8)]],\n"
            "                          constant float      &mu0     "
            "[[buffer(9)]],\n"
            "                          constant float      &omega   "
            "[[buffer(10)]],\n"
            "                          uint2 gid [[thread_position_in_grid]]);\n"
            "\n"
            "Reduce dispatch: 1-D, single threadgroup; the host picks "
            f"`tgsize` (default {_REDUCE_TG}) and dispatches `threadsPerGrid "
            "= tgsize`, `threadsPerThreadgroup = tgsize`. The single TG "
            "must reduce the entire interior into psi_axis[0]. You can "
            "swap to a multi-TG hierarchical reduction or simdgroup ops "
            "as long as psi_axis[0] holds the final max after one "
            "dispatch.\n"
            "\n"
            "Step dispatch: 2-D, threadsPerGrid = (NR, NZ) rounded up to "
            "a multiple of (16, 16); threadsPerThreadgroup = (16, 16, 1) "
            "by default — guard with `if (i >= NR || j >= NZ) return;`. "
            "Boundary cells MUST copy psi_in -> psi_out unchanged. Each "
            "thread MUST update exactly one cell; the host will not "
            "shrink the dispatch.\n"
            "\n"
            "IMPORTANT — threadgroup geometry is set by the host, not the "
            "kernel. The host always picks tg_w = 16 and only ever shrinks "
            "tg_h by halving (16×16 → 16×8 → 16×4 → 16×2 → 16×1) IF the "
            "kernel's [[max_total_threads_per_threadgroup(N)]] attribute "
            "forces a smaller cap. So the only TG shapes you can actually "
            "be dispatched with are (16, 16), (16, 8), (16, 4), (16, 2), "
            "(16, 1). You CANNOT get a (32, 8) or (8, 32) TG by writing "
            "the attribute or by `#define`-ing a tile size.\n"
            "\n"
            "If you do threadgroup-memory tiling for the stencil, your "
            "tile dims MUST equal the dispatched TG dims (e.g. a 16×16 "
            "tile + halo, sized to the default TG). Computing a tile "
            "origin as `tgid.xy * TILE` only matches the dispatch when "
            "TILE equals the TG dims; otherwise tiles overlap (or leave "
            "gaps) and the result is non-deterministic / NaN. Same "
            "constraint for the reduction: its dispatched TG width is "
            f"{_REDUCE_TG} threads (or smaller if you cap it via the "
            "max-threads attribute) — design your reduction around that."
        ),
        kernel_names=["gradshaf_axis_reduce", "gradshaf_step"],
        seed_path=_SEED,
        sizes=[
            TaskSize("65x65_30",    {"nr": 65,  "nz": 65,  "k": 30}),
            TaskSize("257x257_40",  {"nr": 257, "nz": 257, "k": 40}),
            TaskSize("513x513_30",  {"nr": 513, "nz": 513, "k": 30}),
        ],
        held_out_sizes=[
            TaskSize("129x129_35",  {"nr": 129, "nz": 129, "k": 35}),
        ],
    )

    def evaluate_size(self, harness, pipelines, size, chip, n_warmup, n_measure):
        NR = int(size.params["nr"])
        NZ = int(size.params["nz"])
        K = int(size.params["k"])

        dR = float((_RMAX - _RMIN) / (NR - 1))
        dZ = float((_ZMAX - _ZMIN) / (NZ - 1))

        psi0 = _make_init(NR, NZ)               # (NZ, NR) fp32
        nbytes = psi0.nbytes

        bA = harness.buf_from_np(psi0)
        bB = harness.buf_zeros(nbytes)
        bAxis = harness.buf_zeros(4)            # 1 fp32 scalar
        bNR = harness.buf_scalar(NR, np.uint32)
        bNZ = harness.buf_scalar(NZ, np.uint32)
        bRmin = harness.buf_scalar(_RMIN, np.float32)
        bdR = harness.buf_scalar(dR, np.float32)
        bdZ = harness.buf_scalar(dZ, np.float32)
        bPaxis = harness.buf_scalar(_P_AXIS, np.float32)
        bMu0 = harness.buf_scalar(_MU0, np.float32)
        bOmega = harness.buf_scalar(_OMEGA, np.float32)

        pso_reduce = pipelines["gradshaf_axis_reduce"]
        pso_step = pipelines["gradshaf_step"]

        # Reduction: single-TG dispatch. Honour the PSO's max if the LLM's
        # variant declared a tighter cap.
        red_max = int(pso_reduce.maxTotalThreadsPerThreadgroup())
        red_tg = min(_REDUCE_TG, red_max)
        # Round down to a power of 2 for the tree reduction in the seed.
        # (LLM variants that don't need power-of-2 are still served — they
        # just see the chosen tg via threads_per_threadgroup.)
        pow2 = 1
        while pow2 * 2 <= red_tg:
            pow2 *= 2
        red_tg = pow2

        # Step: 2-D, 16x16 default; matches heat2d.
        step_max = int(pso_step.maxTotalThreadsPerThreadgroup())
        tg_w, tg_h = 16, 16
        while tg_w * tg_h > step_max:
            tg_h //= 2
        grid_w = ((NR + tg_w - 1) // tg_w) * tg_w
        grid_h = ((NZ + tg_h - 1) // tg_h) * tg_h

        view_A = harness.np_view(bA, np.float32, NR * NZ)
        view_B = harness.np_view(bB, np.float32, NR * NZ)

        def reset():
            view_A[:] = psi0.ravel()
            view_B[:] = 0.0

        def dispatch(enc):
            # Constant bindings that don't change across the K outer steps.
            for step in range(K):
                in_buf, out_buf = (bA, bB) if step % 2 == 0 else (bB, bA)

                # 1) max-reduction over the interior of `in_buf`.
                enc.setComputePipelineState_(pso_reduce)
                enc.setBuffer_offset_atIndex_(in_buf, 0, 0)
                enc.setBuffer_offset_atIndex_(bAxis, 0, 1)
                enc.setBuffer_offset_atIndex_(bNR, 0, 2)
                enc.setBuffer_offset_atIndex_(bNZ, 0, 3)
                enc.dispatchThreads_threadsPerThreadgroup_(
                    Metal.MTLSizeMake(red_tg, 1, 1),
                    Metal.MTLSizeMake(red_tg, 1, 1),
                )

                # 2) per-cell Δ* + nonlinear source + Jacobi update.
                enc.setComputePipelineState_(pso_step)
                enc.setBuffer_offset_atIndex_(in_buf, 0, 0)
                enc.setBuffer_offset_atIndex_(out_buf, 0, 1)
                enc.setBuffer_offset_atIndex_(bAxis, 0, 2)
                enc.setBuffer_offset_atIndex_(bNR, 0, 3)
                enc.setBuffer_offset_atIndex_(bNZ, 0, 4)
                enc.setBuffer_offset_atIndex_(bRmin, 0, 5)
                enc.setBuffer_offset_atIndex_(bdR, 0, 6)
                enc.setBuffer_offset_atIndex_(bdZ, 0, 7)
                enc.setBuffer_offset_atIndex_(bPaxis, 0, 8)
                enc.setBuffer_offset_atIndex_(bMu0, 0, 9)
                enc.setBuffer_offset_atIndex_(bOmega, 0, 10)
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

        # Final correctness pass: re-run from a clean reset so the buffer
        # holding the answer is deterministic w.r.t. the ping-pong index.
        reset()
        harness.time_dispatch(dispatch)
        final_view = view_B if K % 2 == 1 else view_A
        got = final_view.copy().reshape(NZ, NR)

        expected = _cpu_reference(
            psi0, NR, NZ, K,
            Rmin=_RMIN, Rmax=_RMAX, Zmin=_ZMIN, Zmax=_ZMAX,
            p_axis=_P_AXIS, mu0=_MU0, omega=_OMEGA,
        )
        max_ref = float(np.max(np.abs(expected)))
        err = float(np.max(np.abs(got - expected)))
        # fp32 stencil + reduction over K Picard steps. The reduction is
        # associativity-safe (max is commutative/associative); the stencil
        # is the same operation order CPU↔GPU; drift comes from the
        # nonlinear source's self-feedback through ψ_axis. Heat2d's
        # 1e-4 + 1e-5·max|ref| holds at K=100 there; this task at K≤40
        # comfortably fits the same envelope.
        tol = 1e-4 + 1e-5 * max_ref
        correct = err <= tol

        # BW-bound roofline. Per Picard outer step, unique DRAM traffic
        # at the cache-perfect limit is:
        #   reduce: 4 B/cell (one read of the full field, one scalar write
        #           amortised away)
        #   step:   8 B/cell (read 1 ψ, write 1 ψ; the 4 stencil neighbours
        #           are amortised by L1/L2 reuse, ψ_axis is hot in cache)
        # Total: 12 B/cell per outer step.
        bytes_per_step = 12.0 * NR * NZ
        bytes_total = bytes_per_step * K
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
            extra={"tol": tol, "nr": NR, "nz": NZ, "k": K,
                   "p_axis": _P_AXIS, "mu0": _MU0, "omega": _OMEGA},
        )
