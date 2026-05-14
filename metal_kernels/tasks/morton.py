"""Morton-ordered 3D heat stencil (Plan §R8: bit-permutation memory).

A 7-point Laplacian forward-Euler timestep on a 3D scalar field whose
storage is Z-order (Morton-) interleaved instead of row-major. For each
cell at (x, y, z) ∈ [0, N)^2 the linear buffer index is

    M(x, y, z) = sum_{i=0}^{logN-1} ( x[i]·2^(3i)
                                      + y[i]·2^(3i+1)
                                      + z[i]·2^(3i+2) )

where x[i] is the i-th bit of x and logN = log2(N). N is a power of 2,
M is a bijection onto [0, N^3), and the buffer is exactly N^3 floats.

The candidate writes one ``morton_stencil`` kernel that runs ONE timestep
(7-point Laplacian, forward Euler with stability margin alpha = 0.10 <
1/6). The host iterates n_steps timesteps in one command buffer,
ping-ponging u_in/u_out for accurate end-to-end GPU timing — same
pattern as heat2d and wave3d.

Optimization lever:
  - Morton encode/decode efficiency. The seed uses an O(logN) per-bit
    loop in both directions. Magic-constant bit spreading (PDEP-style)
    compresses this to O(1) constant-time arithmetic; 256-entry per-byte
    lookup tables in constant memory trade register pressure for FMAs.
  - Neighbour-index arithmetic via direct bit-twiddling on the Morton
    index (m_xp = ((m | YZ_MASK) + 1) & X_MASK | (m & YZ_MASK) with
    X_MASK = 0b...001001001, YZ_MASK = ~X_MASK truncated to 3·logN bits)
    avoids the full encode round-trip for the six neighbours.
  - Cache locality of Morton order. Consecutive Morton indices cluster
    spatially: 32 consecutive indices cover a compact 4·2·4 block (for
    logN ≥ 3), so dispatching threads 1-D with tid = Morton index lets
    each simdgroup keep its stencil neighbours in L1/SLC. The seed
    already gets this geometry — the candidate must keep it.

Held-out (N=256): the working set (2 × 256^3 × 4 B ≈ 128 MB) is well
beyond the M1 Pro 24 MB SLC, so the kernel is solidly DRAM-bound and
Morton's cache-locality benefit actually matters here. The in-dist
sizes 32^3, 64^3, 128^3 are SLC-resident, so a candidate that tunes for
pure compute efficiency (skipping the locality argument) can ace them
and silently regress at 256^3 — the same silent-regression failure mode
as fft3d-GPT.

Roofline: BW-bound at 8 B/cell per step (1 read + 1 write; the six
neighbours are amortised by L1/SLC), matching heat2d. The smallest sizes
are SLC-resident and reported fractions may exceed 1 — same caveat as
fft3d 32^3 and lbm 256^2.
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


_SEED = Path(__file__).resolve().parent.parent.parent / "seeds" / "morton.metal"


def _spread_bits3(v: np.ndarray, logN: int) -> np.ndarray:
    """Spread bits: bit i of v -> bit 3i of output. Vectorised; uint32 out."""
    out = np.zeros_like(v, dtype=np.uint32)
    for i in range(logN):
        out |= ((v >> i) & np.uint32(1)) << np.uint32(3 * i)
    return out


def _morton_table(N: int) -> np.ndarray:
    """(N, N, N) uint32: table[z, y, x] = M(x, y, z).

    Vectorised via 3 × 1-D bit-spreading + broadcasting. logN ≤ 8 across
    the size set, so uint32 is comfortably sufficient (24 bits needed).
    """
    logN = int(round(np.log2(N)))
    assert N == 1 << logN, f"N={N} must be a power of 2"
    rng = np.arange(N, dtype=np.uint32)
    spread = _spread_bits3(rng, logN)
    x_s = spread                    # bits at 3i
    y_s = spread << np.uint32(1)    # bits at 3i+1
    z_s = spread << np.uint32(2)    # bits at 3i+2
    return (z_s[:, None, None] | y_s[None, :, None] | x_s[None, None, :]).astype(np.uint32)


def _make_init(N: int) -> np.ndarray:
    """Smooth 3D Gaussian bump centred in the box, zero on boundary, fp32.

    Shape (NZ=N, NY=N, NX=N), C-contiguous. A Gaussian (rather than a
    delta or random field) keeps high-frequency content low so fp32
    rounding noise stays manageable across many steps.
    """
    iz = np.arange(N, dtype=np.float32).reshape(N, 1, 1)
    iy = np.arange(N, dtype=np.float32).reshape(1, N, 1)
    ix = np.arange(N, dtype=np.float32).reshape(1, 1, N)
    c = np.float32((N - 1) / 2.0)
    sigma = np.float32(0.15 * N)
    r2 = (ix - c) ** 2 + (iy - c) ** 2 + (iz - c) ** 2
    u = np.exp(-r2 / (np.float32(2.0) * sigma * sigma)).astype(np.float32)
    u[0, :, :] = 0.0
    u[-1, :, :] = 0.0
    u[:, 0, :] = 0.0
    u[:, -1, :] = 0.0
    u[:, :, 0] = 0.0
    u[:, :, -1] = 0.0
    return np.ascontiguousarray(u)


def _cpu_reference(u0: np.ndarray, alpha: float, n_steps: int) -> np.ndarray:
    """Reference 3D 7-point forward-Euler heat update in row-major.

    Vectorised via numpy slicing. Boundary cells (any face of the cube)
    are copied through unchanged, matching the kernel's Dirichlet rule.
    Returns u^{n_steps} in (NZ, NY, NX) row-major.
    """
    a = np.float32(alpha)
    six = np.float32(6.0)
    u = u0.astype(np.float32, copy=True)
    v = np.empty_like(u)
    for _ in range(n_steps):
        c = u[1:-1, 1:-1, 1:-1]
        v[1:-1, 1:-1, 1:-1] = (
            c + a * (
                u[1:-1, 1:-1, :-2] + u[1:-1, 1:-1, 2:]
                + u[1:-1, :-2, 1:-1] + u[1:-1, 2:, 1:-1]
                + u[:-2, 1:-1, 1:-1] + u[2:, 1:-1, 1:-1]
                - six * c
            )
        )
        v[0, :, :] = u[0, :, :]
        v[-1, :, :] = u[-1, :, :]
        v[:, 0, :] = u[:, 0, :]
        v[:, -1, :] = u[:, -1, :]
        v[:, :, 0] = u[:, :, 0]
        v[:, :, -1] = u[:, :, -1]
        u, v = v, u
    return u


@register_task("morton")
class MortonTask(Task):
    spec = TaskSpec(
        name="morton",
        description=(
            "3D heat-equation stencil with Morton (Z-order) buffer layout. "
            "For each cell at (x, y, z) ∈ [0, N)^3, the linear buffer "
            "index is the bit-interleave\n"
            "  M(x, y, z) = sum_{i=0}^{logN-1} ( x[i]·2^(3i)\n"
            "                                    + y[i]·2^(3i+1)\n"
            "                                    + z[i]·2^(3i+2) )\n"
            "with x[i] the i-th bit of x and logN = log2(N). N is a power "
            "of 2 in every test (both in-distribution and held-out), so M "
            "is a bijection onto [0, N^3) and the buffer is exactly N^3 "
            "floats with no padding.\n\n"
            "One forward-Euler timestep, 7-point Laplacian:\n"
            "  u_new[M(x,y,z)] = u[M(x,y,z)] + alpha * (\n"
            "         u[M(x-1,y,z)] + u[M(x+1,y,z)]\n"
            "       + u[M(x,y-1,z)] + u[M(x,y+1,z)]\n"
            "       + u[M(x,y,z-1)] + u[M(x,y,z+1)]\n"
            "       - 6 u[M(x,y,z)] )\n"
            "Stability requires alpha < 1/6 for the 3D 7-point heat "
            "stencil; the host uses alpha = 0.10. Dirichlet BC: every "
            "cell with x, y, or z in {0, N-1} (a face of the cube) MUST "
            "copy u → u_new unchanged. The initial state has those faces "
            "hard-zero. The host ping-pongs u_in/u_out across n_steps "
            "timesteps in one command buffer for accurate end-to-end GPU "
            "timing.\n\n"
            "Optimization lever, unique to this task:\n"
            "  (a) Morton encode/decode efficiency. The seed uses an "
            "O(logN) per-bit loop. Magic-constant bit spreading "
            "(PDEP-style) is O(1):\n"
            "    uint spread3(uint v) {  // pack 8 bits at stride 3\n"
            "        v = (v | (v << 16)) & 0x030000FFu;\n"
            "        v = (v | (v <<  8)) & 0x0300F00Fu;\n"
            "        v = (v | (v <<  4)) & 0x030C30C3u;\n"
            "        v = (v | (v <<  2)) & 0x09249249u;\n"
            "        return v;\n"
            "    }\n"
            "    uint M(uint x, uint y, uint z) {\n"
            "        return spread3(x) | (spread3(y)<<1) | (spread3(z)<<2);\n"
            "    }\n"
            "  256-entry per-byte lookup tables in constant memory are an "
            "alternative that trades register pressure for arithmetic.\n"
            "  (b) Neighbour-index arithmetic on the Morton index "
            "directly, avoiding the encode round-trip. With\n"
            "    X_MASK = 0x09249249u  // bits 0, 3, 6, ... up to 3·logN\n"
            "    Y_MASK = 0x12492492u  // bits 1, 4, 7, ...\n"
            "    Z_MASK = 0x24924924u  // bits 2, 5, 8, ...\n"
            "  (each truncated to 3·logN bits) one has\n"
            "    m_xp = ((m | (Y_MASK | Z_MASK)) + 1u) & X_MASK\n"
            "             | (m & (Y_MASK | Z_MASK));\n"
            "    m_xm = ((m & X_MASK) - 1u) & X_MASK\n"
            "             | (m & (Y_MASK | Z_MASK));\n"
            "  and analogous formulas for ± y, ± z by rotating the masks.\n"
            "  (c) Cache locality of the Morton traversal. Consecutive "
            "Morton indices cluster spatially: for logN ≥ 3 a 32-thread "
            "simdgroup covers a 4·2·4 block, so its 6 stencil neighbours "
            "reuse L1/SLC heavily. Threads MUST be dispatched 1-D with "
            "tid = Morton index — that is what the seed does and what "
            "the locality argument requires.\n\n"
            "The in-distribution sizes (32, 64, 128) are SLC-resident on "
            "M1 Pro; the held-out 256^3 has a 128 MB ping-pong working "
            "set and is solidly DRAM-bound, so the cache-locality lever "
            "actually matters there. A candidate that wins in-distribution "
            "by pure encode/decode speedups without delivering locality "
            "will reveal that at the held-out size."
        ),
        kernel_signatures=(
            "kernel void morton_stencil(\n"
            "    device const float *u_in   [[buffer(0)]],\n"
            "    device       float *u_out  [[buffer(1)]],\n"
            "    constant uint      &N      [[buffer(2)]],\n"
            "    constant uint      &logN   [[buffer(3)]],\n"
            "    constant float     &alpha  [[buffer(4)]],\n"
            "    uint tid [[thread_position_in_grid]]);\n"
            "\n"
            "Dispatch geometry (host-fixed): 1-D dispatch of N^3 threads "
            "padded up to a multiple of the chosen TG width; the host "
            "picks tg_width = min(256, maxTotalThreadsPerThreadgroup). "
            "Threads MUST early-exit if tid >= N^3.\n\n"
            "Convention: tid is the MORTON INDEX (not a (x,y,z) linear "
            "position). Consecutive threads therefore access consecutive "
            "buffer elements u_in[tid] / u_out[tid] — this is the trait "
            "Morton ordering exists to exploit; the kernel must keep it. "
            "Inside the kernel: decode tid → (x, y, z) for the boundary "
            "check, then compute the Morton indices of the six neighbours "
            "to gather the stencil. logN is provided as a separate "
            "constant so the kernel can iterate exactly logN times "
            "(5/6/7/8 across the size set) without a runtime log2; the "
            "host guarantees N == 1 << logN.\n\n"
            "If you cap the kernel with [[max_total_threads_per_"
            "threadgroup(W)]], place the attribute on the kernel "
            "declaration itself; the host picks tg_width = min(W, 256). "
            "Buffers 0 and 1 are read/write; the host ping-pongs their "
            "roles across timesteps, so do NOT assume u_in and u_out "
            "alias fixed addresses."
        ),
        kernel_names=["morton_stencil"],
        seed_path=_SEED,
        sizes=[
            # 32^3 = 32K cells = 128 KB per buffer (L2-resident).
            # 64^3 = 256K cells = 1 MB (L2/SLC).
            # 128^3 = 2M cells = 8 MB (SLC; 16 MB across ping-pong fits
            # the 24 MB M1 Pro SLC).
            TaskSize("N32_120",  {"N": 32,  "n_steps": 120}),
            TaskSize("N64_60",   {"N": 64,  "n_steps": 60}),
            TaskSize("N128_30",  {"N": 128, "n_steps": 30}),
        ],
        held_out_sizes=[
            # 256^3 = 16M cells = 64 MB per buffer (128 MB across the
            # ping-pong) — solidly DRAM-bound, well beyond M1 Pro SLC.
            # The Morton cache-locality lever actually matters here;
            # candidates that won in-dist via encode/decode speed alone
            # (without keeping the locality-friendly tid=Morton mapping)
            # will reveal that here. Symmetric in spirit to fft3d's
            # held-out 256^3 cube.
            TaskSize("N256_10", {"N": 256, "n_steps": 10}),
        ],
    )

    def evaluate_size(self, harness, pipelines, size, chip, n_warmup, n_measure):
        N = int(size.params["N"])
        n_steps = int(size.params["n_steps"])
        alpha = 0.10

        logN = int(round(np.log2(N)))
        if N != 1 << logN:
            raise ValueError(f"morton: N={N} must be a power of 2")

        # ---- inputs --------------------------------------------------
        # u0_row is the canonical (NZ, NY, NX) row-major view used by the
        # CPU reference. u0_morton is the same data permuted to Morton
        # order; that is what the kernel sees.
        u0_row = _make_init(N)
        mtable = _morton_table(N)                              # (N,N,N) uint32
        flat_mtable = mtable.ravel().astype(np.int64)          # safe to scatter
        u0_morton = np.empty(N ** 3, dtype=np.float32)
        u0_morton[flat_mtable] = u0_row.ravel()                # M(x,y,z) ← u0[z,y,x]
        u0_morton = np.ascontiguousarray(u0_morton)
        nbytes = u0_morton.nbytes

        # ---- buffers + pipelines -------------------------------------
        bA = harness.buf_from_np(u0_morton)
        bB = harness.buf_zeros(nbytes)
        bN = harness.buf_scalar(N, np.uint32)
        bLogN = harness.buf_scalar(logN, np.uint32)
        bAlpha = harness.buf_scalar(alpha, np.float32)

        pso = pipelines["morton_stencil"]
        tg = min(256, int(pso.maxTotalThreadsPerThreadgroup()))
        total = N * N * N
        grid = ((total + tg - 1) // tg) * tg
        size_grid = Metal.MTLSizeMake(grid, 1, 1)
        size_tg = Metal.MTLSizeMake(tg, 1, 1)

        view_A = harness.np_view(bA, np.float32, total)
        view_B = harness.np_view(bB, np.float32, total)

        def reset():
            view_A[:] = u0_morton
            view_B[:] = 0.0

        def dispatch(encoder):
            encoder.setComputePipelineState_(pso)
            encoder.setBuffer_offset_atIndex_(bN, 0, 2)
            encoder.setBuffer_offset_atIndex_(bLogN, 0, 3)
            encoder.setBuffer_offset_atIndex_(bAlpha, 0, 4)
            for step in range(n_steps):
                if step % 2 == 0:
                    in_buf, out_buf = bA, bB
                else:
                    in_buf, out_buf = bB, bA
                encoder.setBuffer_offset_atIndex_(in_buf, 0, 0)
                encoder.setBuffer_offset_atIndex_(out_buf, 0, 1)
                encoder.dispatchThreads_threadsPerThreadgroup_(size_grid, size_tg)

        # ---- warmup + measure ----------------------------------------
        for _ in range(n_warmup):
            reset()
            harness.time_dispatch(dispatch)
        samples = []
        for _ in range(n_measure):
            reset()
            samples.append(harness.time_dispatch(dispatch))
        gpu_s = float(np.median(samples))

        # ---- correctness ---------------------------------------------
        reset()
        harness.time_dispatch(dispatch)
        final_view = view_A if n_steps % 2 == 0 else view_B
        got_morton = final_view.copy()
        # Un-permute Morton → row-major for comparison with the CPU ref.
        got_row = got_morton[flat_mtable].reshape(N, N, N)

        expected = _cpu_reference(u0_row, alpha, n_steps)
        max_ref = float(np.max(np.abs(expected)))
        err = float(np.max(np.abs(got_row - expected)))
        # fp32 forward-Euler heat over n_steps accumulates O(eps · n_steps
        # · max|u|) rounding noise. The Gaussian peak is 1.0 so |max_ref|
        # ≤ 1; 1e-4 absolute is comfortably above the observed drift
        # while still catching real bugs (a wrong neighbour index would
        # produce O(0.1)-scale disagreement once diffusion smooths the
        # bump).
        tol = 1e-4 + 1e-5 * max_ref
        correct = err <= tol

        # ---- roofline -------------------------------------------------
        # BW-bound at 8 B/cell per step (1 read center + 1 write; the
        # six neighbours are amortised by L1/SLC at the cache-perfect
        # limit). Same convention as heat2d.
        bytes_per_step = 8.0 * N * N * N
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
            extra={"tol": tol, "N": N, "logN": logN,
                   "n_steps": n_steps, "alpha": alpha},
        )
