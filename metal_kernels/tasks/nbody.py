"""All-pairs gravitational N-body with leapfrog integrator.

Compute-bound: ~20 FLOPs per pair-interaction. Per step we do N*N pair
interactions, so total work scales O(N^2 * n_steps).

Candidate writes one ``nbody_step`` kernel that reads ``pos_in/vel_in``
and writes ``pos_out/vel_out``. The host ping-pongs between two
position+velocity buffer pairs across ``n_steps`` iterations, all packed
into a single command buffer for accurate timing.
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


_SEED = Path(__file__).resolve().parent.parent.parent / "seeds" / "nbody.metal"


def _cpu_reference(pos: np.ndarray, vel: np.ndarray, mass: np.ndarray,
                   dt: float, eps: float, G: float, n_steps: int):
    """Reference all-pairs leapfrog, fp32, vectorized via numpy.

    For each step:
        a_i = G * sum_j m_j * (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)
        v <- v + a * dt
        r <- r + v * dt
    """
    pos = pos.astype(np.float32, copy=True)
    vel = vel.astype(np.float32, copy=True)
    mass = mass.astype(np.float32, copy=False)
    eps2 = np.float32(eps * eps)
    for _ in range(n_steps):
        # diff[i,j,k] = pos[j,k] - pos[i,k]
        diff = pos[None, :, :] - pos[:, None, :]
        r2 = np.einsum("ijk,ijk->ij", diff, diff) + eps2
        inv_r3 = (r2 ** -1.5).astype(np.float32)
        # Zero self-interaction is auto-handled by softening eps.
        a = G * np.einsum("ij,j,ijk->ik", inv_r3, mass, diff).astype(np.float32)
        vel = vel + a * np.float32(dt)
        pos = pos + vel * np.float32(dt)
    return pos, vel


def _make_init(n: int, seed: int = 0xCAFE):
    rng = np.random.default_rng(seed)
    # Plummer-ish: positions in unit cube, masses uniform.
    pos3 = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    vel3 = rng.uniform(-0.01, 0.01, size=(n, 3)).astype(np.float32)
    mass = rng.uniform(0.5, 1.5, size=(n,)).astype(np.float32)
    # Pad to float4 layout for Metal (.w = 0).
    pos4 = np.zeros((n, 4), dtype=np.float32)
    pos4[:, :3] = pos3
    vel4 = np.zeros((n, 4), dtype=np.float32)
    vel4[:, :3] = vel3
    return pos3, vel3, mass, pos4, vel4


@register_task("nbody")
class NBodyTask(Task):
    spec = TaskSpec(
        name="nbody",
        description=(
            "All-pairs gravitational N-body with leapfrog integration. "
            "For each body i:\n"
            "  a_i = G * sum_{j} m_j (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)\n"
            "  v_new = v + a * dt\n"
            "  r_new = r + v_new * dt\n"
            "Self-interaction is masked by the softening epsilon (no special "
            "case needed). Positions/velocities are packed as float4 with "
            ".xyz holding the data and .w padding. Masses are float[N]."
        ),
        kernel_signatures=(
            "kernel void nbody_step(device const float4 *pos_in  [[buffer(0)]],\n"
            "                       device       float4 *pos_out [[buffer(1)]],\n"
            "                       device const float4 *vel_in  [[buffer(2)]],\n"
            "                       device       float4 *vel_out [[buffer(3)]],\n"
            "                       device const float  *mass    [[buffer(4)]],\n"
            "                       constant uint        &N      [[buffer(5)]],\n"
            "                       constant float       &dt     [[buffer(6)]],\n"
            "                       constant float       &eps    [[buffer(7)]],\n"
            "                       constant float       &G      [[buffer(8)]],\n"
            "                       uint i [[thread_position_in_grid]]);\n"
            "\n"
            "Threads are dispatched 1-D, one per body — guard with `if "
            "(i >= N) return;`. Each thread MUST update exactly one body; "
            "the host will not shrink the dispatch if you process multiple "
            "bodies per thread, so extra threads just idle. The host "
            "ping-pongs (pos_in, pos_out) and (vel_in, vel_out) buffer "
            "pairs each step."
        ),
        kernel_names=["nbody_step"],
        seed_path=_SEED,
        sizes=[
            TaskSize("256_10",  {"n": 256,  "n_steps": 10}),
            TaskSize("1024_10", {"n": 1024, "n_steps": 10}),
            TaskSize("2048_10", {"n": 2048, "n_steps": 10}),
        ],
        held_out_sizes=[
            TaskSize("512_10", {"n": 512, "n_steps": 10}),
        ],
    )

    def evaluate_size(self, harness, pipelines, size, chip, n_warmup, n_measure):
        n = int(size.params["n"])
        n_steps = int(size.params["n_steps"])
        dt = 1e-3
        eps = 0.05
        G = 1.0

        pos3_0, vel3_0, mass, pos4_0, vel4_0 = _make_init(n)

        b_posA = harness.buf_from_np(pos4_0)
        b_posB = harness.buf_zeros(pos4_0.nbytes)
        b_velA = harness.buf_from_np(vel4_0)
        b_velB = harness.buf_zeros(vel4_0.nbytes)
        b_mass = harness.buf_from_np(mass)
        b_N    = harness.buf_scalar(n, np.uint32)
        b_dt   = harness.buf_scalar(dt, np.float32)
        b_eps  = harness.buf_scalar(eps, np.float32)
        b_G    = harness.buf_scalar(G, np.float32)

        pso = pipelines["nbody_step"]
        max_tg = int(pso.maxTotalThreadsPerThreadgroup())
        tew = int(pso.threadExecutionWidth())
        tg_w = min(max_tg, max(tew, 128))
        grid_w = ((n + tg_w - 1) // tg_w) * tg_w

        view_posA = harness.np_view(b_posA, np.float32, n * 4)
        view_velA = harness.np_view(b_velA, np.float32, n * 4)
        view_posB = harness.np_view(b_posB, np.float32, n * 4)
        view_velB = harness.np_view(b_velB, np.float32, n * 4)

        def reset():
            view_posA[:] = pos4_0.ravel()
            view_velA[:] = vel4_0.ravel()
            view_posB[:] = 0.0
            view_velB[:] = 0.0

        def dispatch(enc):
            enc.setComputePipelineState_(pso)
            enc.setBuffer_offset_atIndex_(b_mass, 0, 4)
            enc.setBuffer_offset_atIndex_(b_N, 0, 5)
            enc.setBuffer_offset_atIndex_(b_dt, 0, 6)
            enc.setBuffer_offset_atIndex_(b_eps, 0, 7)
            enc.setBuffer_offset_atIndex_(b_G, 0, 8)
            for step in range(n_steps):
                if step % 2 == 0:
                    pin, pout = b_posA, b_posB
                    vin, vout = b_velA, b_velB
                else:
                    pin, pout = b_posB, b_posA
                    vin, vout = b_velB, b_velA
                enc.setBuffer_offset_atIndex_(pin, 0, 0)
                enc.setBuffer_offset_atIndex_(pout, 0, 1)
                enc.setBuffer_offset_atIndex_(vin, 0, 2)
                enc.setBuffer_offset_atIndex_(vout, 0, 3)
                enc.dispatchThreads_threadsPerThreadgroup_(
                    Metal.MTLSizeMake(grid_w, 1, 1),
                    Metal.MTLSizeMake(tg_w, 1, 1),
                )

        for _ in range(n_warmup):
            reset()
            harness.time_dispatch(dispatch)
        samples = []
        for _ in range(n_measure):
            reset()
            samples.append(harness.time_dispatch(dispatch))
        gpu_s = float(np.median(samples))

        # Final correctness pass
        reset()
        harness.time_dispatch(dispatch)
        final_pos_view = view_posA if n_steps % 2 == 0 else view_posB
        got_pos = final_pos_view.copy().reshape(n, 4)[:, :3]

        ref_pos, _ = _cpu_reference(pos3_0, vel3_0, mass, dt, eps, G, n_steps)
        # Position drift tolerance: integrators differ by ~O(dt^2 * N * n_steps)
        # in the perturbation; for n_steps=10, dt=1e-3, this is small but
        # nontrivial. Use mixed absolute+relative.
        max_pos = float(np.max(np.abs(ref_pos)))
        err = float(np.max(np.abs(got_pos - ref_pos)))
        tol = 1e-3 + 1e-3 * max_pos
        correct = err <= tol

        # FLOP count per pair: 3 sub, 3 mul, 2 add (dot), 1 add (eps2),
        # 1 mul, 1 mul, 1 rsqrt (~4), 3 mul, 3 mul, 3 add ≈ 20 FLOPs.
        flops_per_pair = 20.0
        total_flops = flops_per_pair * (n * n) * n_steps
        achieved = gflops(total_flops, gpu_s)
        ceiling = float(chip.peak_fp32_gflops)
        return SizeResult(
            size_label=size.label,
            correct=correct,
            error_value=err,
            error_kind="max_abs_pos",
            gpu_seconds=gpu_s,
            achieved=achieved,
            achieved_unit="GFLOPS",
            ceiling=ceiling,
            ceiling_unit="GFLOPS",
            fraction_of_ceiling=achieved / ceiling if ceiling > 0 else 0.0,
            extra={"tol": tol, "n": n, "n_steps": n_steps,
                   "dt": dt, "eps": eps, "G": G},
        )
