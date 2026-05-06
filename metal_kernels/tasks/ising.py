"""2D Ising model, checkerboard Metropolis Monte Carlo.

Spins sigma_{i,j} in {-1, +1} on a periodic NX x NY lattice; Hamiltonian
H = -J sum_<ij> sigma_i sigma_j with J = 1. One Metropolis attempt at
site (i, j):

    h    = sigma_left + sigma_right + sigma_down + sigma_up   in {-4,-2,0,2,4}
    dE   = 2 J sigma h                                         in {-8,-4,0,4,8}
    accept iff u < min(1, exp(-beta dE))

A "sweep" is one red sub-pass (sites where (i+j) is even) followed by
one black sub-pass; within each sub-pass all updates are independent
because the neighbours of a color-c site are all color (1-c), so the
sub-pass can be evaluated in parallel.

The benchmark uses a *bit-exact* CPU reference. Two design choices make
this work:

1) The acceptance table p_accept[5] = {1,1,1, exp(-4*beta), exp(-8*beta)}
   is computed once on the host in fp32 and shared by both the GPU
   kernel and the numpy reference, so neither side calls exp().

2) The PRNG is a counter-based Murmur3-fmix32 hash that's identical
   between Metal and numpy:
       x  = seed + step_idx * 0x9E3779B9
       x  = fmix32(x)
       x  = fmix32(x ^ site_idx)
       u  = float(x >> 8) * 2^-24             # 24-bit uniform [0, 1)
   Integer-to-float of a 24-bit value plus multiply by 2^-24 is exact
   in fp32 on every platform; both sides therefore compute exactly the
   same uniform u.

With both pieces fixed, the GPU's per-site decision is a deterministic
function of the host's input bytes, so the verification reduces to a
byte-for-byte equality on the spin array after the dispatch sequence.

Storage is `int8` (one byte per spin), which yields the optimistic
2 B/site/sweep DRAM-traffic ceiling (one read + one write per sweep at
the cache-perfect limit; the four neighbour reads of a checkerboard
sub-pass are fully amortised by the memory hierarchy across the two
sub-passes that make up a sweep).
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


_SEED = Path(__file__).resolve().parent.parent.parent / "seeds" / "ising.metal"

# Beta = 0.42, just below the Onsager critical point beta_c ~ 0.4407.
# Slow-but-not-too-slow mixing; non-trivial cluster structure that
# amplifies any algorithmic bug into a visible disagreement.
_BETA = 0.42
_RNG_SEED = np.uint32(0xC0FFEE01)
_INIT_SEED = 0xBADC0DE


def _make_init(nx: int, ny: int) -> np.ndarray:
    """Random +/-1 spins, deterministic from `_INIT_SEED`."""
    rng = np.random.default_rng(_INIT_SEED)
    bits = rng.integers(0, 2, size=(ny, nx), dtype=np.int8)
    return np.ascontiguousarray((2 * bits - 1).astype(np.int8))


def _p_accept_table(beta: float) -> np.ndarray:
    """p_accept[(s*h + 4) // 2] for s*h in {-4,-2,0,2,4}.

    Indices 0,1,2 (s*h <= 0  →  dE <= 0) accept unconditionally; indices
    3,4 (s*h = 2,4  →  dE = 4,8) accept with exp(-4 beta), exp(-8 beta).
    Computed once in fp32 and shared by GPU kernel and CPU reference so
    neither side has to call exp().
    """
    return np.array([
        1.0,
        1.0,
        1.0,
        np.exp(-4.0 * beta),
        np.exp(-8.0 * beta),
    ], dtype=np.float32)


def _mix32_np(x: np.ndarray) -> np.ndarray:
    """fmix32 (Murmur3 finalizer), element-wise on a uint32 array.

    Mirrors the GPU's `mix32`. We promote to uint64 for the multiplies
    (numpy uint32 * uint32 raises an OverflowError at large values) and
    re-mask back to 32 bits.
    """
    M1 = np.uint64(0x85EBCA6B)
    M2 = np.uint64(0xC2B2AE35)
    MASK = np.uint64(0xFFFFFFFF)
    y = x.astype(np.uint64)
    y = ((y ^ (y >> np.uint64(16))) * M1) & MASK
    y = ((y ^ (y >> np.uint64(13))) * M2) & MASK
    y = (y ^ (y >> np.uint64(16))) & MASK
    return y.astype(np.uint32)


def _rand_u32_np(seed: np.uint32, step_idx: np.ndarray,
                 site_idx: np.ndarray) -> np.ndarray:
    """Counter-based hash matching the GPU's `rand_u32`.

    `seed` is a scalar uint32; `step_idx` and `site_idx` are uint32
    arrays of the same shape (broadcasting is done by the caller).
    """
    GR = np.uint64(0x9E3779B9)
    MASK = np.uint64(0xFFFFFFFF)
    s = (np.uint64(seed) + step_idx.astype(np.uint64) * GR) & MASK
    x = _mix32_np(s.astype(np.uint32))
    x = _mix32_np((x.astype(np.uint64) ^ site_idx.astype(np.uint64)).astype(np.uint32))
    return x


def _bits_to_uniform_fp32(bits: np.ndarray) -> np.ndarray:
    """Convert uint32 hash bits to a 24-bit uniform in [0, 1), exactly
    matching the GPU's `float(bits >> 8) * (1.0f / 16777216.0f)`.
    """
    top24 = (bits >> np.uint32(8)).astype(np.float32)
    return top24 * np.float32(1.0 / 16777216.0)


def _cpu_reference(spins0: np.ndarray, p_accept: np.ndarray,
                   n_sweeps: int, seed: np.uint32) -> np.ndarray:
    """Bit-exact mirror of the GPU dispatch sequence.

    Within each sub-pass the GPU updates all color-c sites in parallel
    using the same set of neighbour values (those of the other color),
    so the order of within-sub-pass updates does not matter. We
    therefore vectorise an entire sub-pass with numpy.
    """
    NY, NX = spins0.shape
    spins = spins0.copy()

    # Color masks (precomputed once).
    j_idx = np.arange(NY, dtype=np.uint32)[:, None]
    i_idx = np.arange(NX, dtype=np.uint32)[None, :]
    site_idx = (j_idx * np.uint32(NX) + i_idx).astype(np.uint32)
    parity = ((j_idx + i_idx) & np.uint32(1)).astype(np.uint32)  # 0 or 1

    for sweep in range(n_sweeps):
        for color in (0, 1):
            step_idx = np.uint32(2 * sweep + color)
            mask = parity == np.uint32(color)

            # Periodic neighbour spins. Sign convention:
            #   sl[j, i] = spins[j, (i - 1) mod NX]   <=>  np.roll(..., +1, axis=1)
            sl = np.roll(spins, +1, axis=1)
            sr = np.roll(spins, -1, axis=1)
            sd = np.roll(spins, +1, axis=0)
            su = np.roll(spins, -1, axis=0)
            h = (sl.astype(np.int32) + sr.astype(np.int32)
                 + sd.astype(np.int32) + su.astype(np.int32))
            prod = spins.astype(np.int32) * h           # in {-4,-2,0,2,4}
            idx = ((prod + 4) // 2).astype(np.int32)    # in {0..4}
            pa = p_accept[idx]                          # fp32

            step_arr = np.full(spins.shape, step_idx, dtype=np.uint32)
            bits = _rand_u32_np(seed, step_arr, site_idx)
            u = _bits_to_uniform_fp32(bits)

            flip = mask & (u < pa)
            spins = np.where(flip, -spins, spins).astype(np.int8)
    return spins


@register_task("ising")
class Ising2DTask(Task):
    spec = TaskSpec(
        name="ising",
        description=(
            "2D Ising model with checkerboard Metropolis updates and "
            "periodic boundary conditions. Spins are int8 in {-1, +1} "
            "stored row-major as `device char *spins[NY*NX]`.\n\n"
            "One sub-pass updates one color of the checkerboard:\n"
            "  color = step_idx & 1   (0 = (i+j) even, 1 = (i+j) odd)\n"
            "The host dispatches this kernel 2 * n_sweeps times with "
            "step_idx = 0, 1, 2, ... so each full sweep is one red pass "
            "followed by one black pass. Within a sub-pass all updates "
            "are independent (the neighbours of a color-c site are all "
            "color 1-c, untouched).\n\n"
            "For each color-matching site (i, j):\n"
            "  h = spins[j,(i-1)%NX] + spins[j,(i+1)%NX]\n"
            "    + spins[(j-1)%NY,i] + spins[(j+1)%NY,i]      in {-4,-2,0,2,4}\n"
            "  prod = spins[j,i] * h                          in {-4,-2,0,2,4}\n"
            "  pa = p_accept[(prod + 4) / 2]                  fp32\n"
            "  draw uniform u in [0, 1) via the prescribed RNG\n"
            "  if u < pa: spins[j,i] = -spins[j,i]\n\n"
            "RNG (must be reproduced bit-exactly):\n"
            "  inline uint mix32(uint x) {\n"
            "      x = (x ^ (x >> 16)) * 0x85EBCA6Bu;\n"
            "      x = (x ^ (x >> 13)) * 0xC2B2AE35u;\n"
            "      return x ^ (x >> 16);\n"
            "  }\n"
            "  uint x = seed + step_idx * 0x9E3779B9u;\n"
            "  x = mix32(x);\n"
            "  x = mix32(x ^ site_idx);            // site_idx = j * NX + i\n"
            "  float u = float(x >> 8) * (1.0f / 16777216.0f);\n"
            "Both the integer-to-float conversion and the multiply by\n"
            "2^-24 are exact in fp32, so candidate kernels MUST use this\n"
            "exact formula (or a provably-equivalent rearrangement).\n"
            "The acceptance table p_accept[5] is precomputed by the host\n"
            "in fp32 and read from buffer 1; do NOT call exp() on the GPU.\n"
        ),
        kernel_signatures=(
            "kernel void ising_step(device       char  *spins    [[buffer(0)]],\n"
            "                       device const float *p_accept [[buffer(1)]],\n"
            "                       constant uint  &NX           [[buffer(2)]],\n"
            "                       constant uint  &NY           [[buffer(3)]],\n"
            "                       constant uint  &step_idx     [[buffer(4)]],\n"
            "                       constant uint  &seed         [[buffer(5)]],\n"
            "                       uint2 gid [[thread_position_in_grid]]);\n"
            "\n"
            "Grid is dispatched 2-D as `threadsPerGrid = (NX, NY)`, one "
            "thread per lattice site; guard with `if (i >= NX || j >= NY) "
            "return;`. Threads of the wrong color (`(i+j) & 1 != color`) "
            "MUST NOT mutate spins[j*NX + i] — they may early-exit or take "
            "the read path with a predicated write. The host will not "
            "shrink the dispatch if you process multiple sites per thread, "
            "so any reorganisation must keep the (NX, NY) grid shape and "
            "the bit-exact RNG/acceptance formula above."
        ),
        kernel_names=["ising_step"],
        seed_path=_SEED,
        sizes=[
            # Three regimes: 256^2 fits comfortably in L2 and is launch-
            # overhead/RNG-bound; 1024^2 (~1 MB working set) sits between
            # cache and DRAM; 2048^2 (4 MB) is solidly DRAM-bound on
            # M-class chips. Sweep counts shrink with size to keep the
            # CPU reference (vectorised numpy) under a few seconds.
            TaskSize("256x256_100",  {"nx": 256,  "ny": 256,  "n_sweeps": 100}),
            TaskSize("1024x1024_50", {"nx": 1024, "ny": 1024, "n_sweeps": 50}),
            TaskSize("2048x2048_25", {"nx": 2048, "ny": 2048, "n_sweeps": 25}),
        ],
        held_out_sizes=[
            TaskSize("1536x1536_40", {"nx": 1536, "ny": 1536, "n_sweeps": 40}),
        ],
    )

    def evaluate_size(self, harness, pipelines, size, chip, n_warmup, n_measure):
        nx = int(size.params["nx"])
        ny = int(size.params["ny"])
        n_sweeps = int(size.params["n_sweeps"])
        beta = _BETA

        spins0 = _make_init(nx, ny)                # (NY, NX) int8
        p_acc  = _p_accept_table(beta)             # (5,) fp32

        # The GPU and CPU reference must see the *same fp32 bytes* in
        # p_accept (so neither calls exp()). The host computes the table
        # once and writes it into a shared MTLBuffer.
        b_spins = harness.buf_from_np(spins0)
        b_p_acc = harness.buf_from_np(p_acc)
        b_NX    = harness.buf_scalar(nx, np.uint32)
        b_NY    = harness.buf_scalar(ny, np.uint32)
        b_seed  = harness.buf_scalar(int(_RNG_SEED), np.uint32)

        # Pre-allocate a small uint32 buffer holding [0, 1, 2, ..., 2*n_sweeps - 1]
        # so that each per-step dispatch sees its own unique step_idx by
        # binding buffer 4 with offset = step * 4.
        step_indices = np.arange(2 * n_sweeps, dtype=np.uint32)
        b_step = harness.buf_from_np(step_indices)

        pso = pipelines["ising_step"]
        max_tg = int(pso.maxTotalThreadsPerThreadgroup())
        # 16x16 = 256 threads; matches lbm/heat2d. Falls back if a high-
        # register-pressure variant tightens the per-PSO cap.
        tg_w, tg_h = 16, 16
        while tg_w * tg_h > max_tg:
            tg_h //= 2
        grid_w = ((nx + tg_w - 1) // tg_w) * tg_w
        grid_h = ((ny + tg_h - 1) // tg_h) * tg_h

        view_spins = harness.np_view(b_spins, np.int8, nx * ny)

        def reset():
            view_spins[:] = spins0.ravel()

        def dispatch(enc):
            enc.setComputePipelineState_(pso)
            enc.setBuffer_offset_atIndex_(b_spins, 0, 0)
            enc.setBuffer_offset_atIndex_(b_p_acc, 0, 1)
            enc.setBuffer_offset_atIndex_(b_NX, 0, 2)
            enc.setBuffer_offset_atIndex_(b_NY, 0, 3)
            enc.setBuffer_offset_atIndex_(b_seed, 0, 5)
            for step in range(2 * n_sweeps):
                enc.setBuffer_offset_atIndex_(b_step, step * 4, 4)
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

        # Final correctness pass: the GPU is bit-exactly deterministic
        # given (spins0, p_accept bytes, seed, step_idx sequence), so we
        # demand a byte-for-byte match against the numpy reference.
        reset()
        harness.time_dispatch(dispatch)
        got = view_spins.copy().reshape(ny, nx)

        expected = _cpu_reference(spins0, p_acc, n_sweeps, _RNG_SEED)

        # Hamming distance (sites where the spin disagrees) — for a
        # correct kernel this must be exactly zero.
        diff_count = int(np.sum(got != expected))
        correct = (diff_count == 0)

        # BW-bound roofline. One sweep = two sub-passes; each site is
        # written once per sweep (in its own color sub-pass), and read
        # both as itself (1 B) and as a neighbour (4 B from the other-
        # color sub-pass). At the cache-perfect limit the four neighbour
        # reads are amortised by L1/L2 reuse, leaving 1 read + 1 write
        # = 2 B/site/sweep of unique DRAM traffic.
        bytes_per_sweep = 2.0 * nx * ny
        bytes_total = bytes_per_sweep * n_sweeps
        achieved = gb_per_s(bytes_total, gpu_s)
        ceiling = float(chip.peak_bw_gb_s)
        return SizeResult(
            size_label=size.label,
            correct=correct,
            error_value=float(diff_count),
            error_kind="spin_disagreements",
            gpu_seconds=gpu_s,
            achieved=achieved,
            achieved_unit="GB/s (effective, 2 B/site/sweep)",
            ceiling=ceiling,
            ceiling_unit="GB/s",
            fraction_of_ceiling=achieved / ceiling if ceiling > 0 else 0.0,
            extra={
                "nx": nx, "ny": ny, "n_sweeps": n_sweeps, "beta": beta,
                "rng_seed": int(_RNG_SEED),
            },
        )
