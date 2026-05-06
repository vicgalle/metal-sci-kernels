"""Hamiltonian Monte Carlo on a Gaussian target.

Many independent chains run in parallel, one thread per chain. The
target is a multivariate Gaussian with mean 0 and covariance Sigma:

    U(q) = (1/2) q^T A q,  A = Sigma^{-1}
    grad U(q) = A q

Per HMC step (one kernel dispatch):

    p ~ N(0, I)             # Box-Muller from per-chain hash RNG
    q_old <- q
    H_old = (1/2) q^T A q + (1/2) p^T p
    leapfrog L steps with step size eps
    H_new = (1/2) q^T A q + (1/2) p^T p
    accept iff log(u_acc) < -(H_new - H_old)

Verification is statistical, not bit-exact. Box-Muller and the
leapfrog mat-vec involve sqrt/log/cos/sin and many fp32 reductions
whose ordering is implementation-dependent, so two correct kernels can
produce slightly different chains. We instead require the empirical
sample distribution after T HMC steps to match the target Gaussian:

  - per-dim |empirical_mean| < tol_mean * sqrt(diag(Sigma))
  - ‖cov_emp - Sigma‖_F / ‖Sigma‖_F < tol_cov
  - acceptance rate in [0.45, 0.99]

With K >= 1024 IID chains each running T = 200 HMC iterations from
overdispersed q_0 ~ N(0, 4 Sigma), the MC standard error of the
sample mean is ~ sigma/sqrt(K) ≈ 0.03 sigma; the chosen tolerances
sit several sigma above that, so a correct kernel passes
deterministically while any sign / leapfrog / accept-rule bug
produces order-1 failures.
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


_SEED = Path(__file__).resolve().parent.parent.parent / "seeds" / "hmc.metal"

# Fixed hyperparameters across sizes.
#
# (eps, L) are tuned to avoid resonances on the anisotropic target Sigma
# whose eigenvalues span [0.5, 2.0] geometrically. With eps * L = 2.0
# the per-step autocorrelation cos(omega * T_traj) sits well away from
# +/-1 for every mode (the slow mode rotates ~80 degrees, the fast mode
# ~160 degrees); CPU calibration at d in {8,16,24,32}, K=1024, T=200
# yields cov_fro_err in [0.11, 0.18] -- comfortably under tol_cov=0.30.
# Acceptance is high (~0.99) because the integrator is accurate at this
# step size; the upper accept-window bound is set just below 1.0 so
# "always accept" bugs still trip it.
_EPS = 0.20
_L = 10
_T_HMC = 200
_RNG_SEED = np.uint32(0xCAFEBEEF)
_TARGET_SEED = 0xC0FFEE     # for the random rotation of Sigma
_INIT_SEED = 0xBADC0DE      # for q_0 generation (numpy RNG, not the GPU hash)


def _make_target(d: int):
    """Return (Sigma, A=Sigma^{-1}) as fp32 (d, d) matrices.

    Eigenvalues of Sigma are geomspace(0.5, 2.0), giving condition
    number 4 — anisotropic enough that single-coordinate Metropolis
    would mix poorly while leapfrog HMC handles it gracefully — and a
    deterministic random orthogonal rotation Q.
    """
    rng = np.random.default_rng(_TARGET_SEED)
    M = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(M)
    eigs = np.geomspace(0.5, 2.0, d)
    Sigma = (Q * eigs) @ Q.T          # Q diag(eigs) Q^T
    A = (Q * (1.0 / eigs)) @ Q.T
    # Symmetrise to clean up any fp drift from the multiply.
    Sigma = 0.5 * (Sigma + Sigma.T)
    A = 0.5 * (A + A.T)
    return Sigma.astype(np.float32), A.astype(np.float32)


def _make_init_q(K: int, Sigma: np.ndarray, overdisp: float = 2.0) -> np.ndarray:
    """q_0 ~ N(0, overdisp^2 * Sigma) per chain. Numpy RNG (not the
    GPU hash); reproducible from `_INIT_SEED`."""
    d = Sigma.shape[0]
    rng = np.random.default_rng(_INIT_SEED)
    L_chol = np.linalg.cholesky(Sigma).astype(np.float32)
    z = rng.standard_normal(size=(K, d)).astype(np.float32)
    return np.ascontiguousarray((overdisp * (z @ L_chol.T)).astype(np.float32))


@register_task("hmc")
class HMCTask(Task):
    spec = TaskSpec(
        name="hmc",
        description=(
            "Hamiltonian Monte Carlo on a multivariate Gaussian target with "
            "mean 0 and precision matrix A = Sigma^{-1} (provided as a "
            "(d, d) row-major float32 buffer). One thread per chain; many "
            "chains run in parallel.\n\n"
            "Per HMC step (one dispatch):\n"
            "  1) p ~ N(0, I): for each pair (i, i+1) in 0..d-1 step 2,\n"
            "     draw two uniforms u1, u2 in [0, 1) via the prescribed RNG\n"
            "     (counters base_counter + i and base_counter + i + 1, where\n"
            "     base_counter = hmc_step_idx * (d + 1)) and apply Box-Muller:\n"
            "        u1 = max(u1, 1e-7);  r = sqrt(-2 * log(u1));\n"
            "        angle = 2 pi * u2;\n"
            "        p[i]   = r * cos(angle);\n"
            "        p[i+1] = r * sin(angle);   // skip if i+1 >= d (d is even).\n"
            "  2) Save q_old = q. Compute force = A q;\n"
            "     U_old = 0.5 * dot(q, force); K_old = 0.5 * dot(p, p).\n"
            "  3) Leapfrog with eps:\n"
            "        p   -= (eps/2) * force            // initial half-kick\n"
            "        for l = 0..L-1:\n"
            "            q   += eps * p                // drift\n"
            "            force = A q                   // recompute force at new q\n"
            "            scale = (l + 1 == L) ? (eps/2) : eps\n"
            "            p   -= scale * force          // kick\n"
            "  4) U_new = 0.5 * dot(q, force) [reusing the final force];\n"
            "     K_new = 0.5 * dot(p, p);\n"
            "     dH = (U_new + K_new) - (U_old + K_old).\n"
            "  5) Draw uniform u_acc with counter base_counter + d.\n"
            "     accept = isfinite(dH) AND log(max(u_acc, 1e-30)) < -dH.\n"
            "     Write q if accept else q_old to q_out[chain_idx * d + i];\n"
            "     if accept, accept_cnt[chain_idx] += 1.\n\n"
            "RNG (must be reproduced bit-exactly):\n"
            "  inline uint mix32(uint x) {\n"
            "      x = (x ^ (x >> 16)) * 0x85EBCA6Bu;\n"
            "      x = (x ^ (x >> 13)) * 0xC2B2AE35u;\n"
            "      return x ^ (x >> 16);\n"
            "  }\n"
            "  uint x = seed + chain_idx * 0x9E3779B9u;\n"
            "  x = mix32(x ^ counter);\n"
            "  x = mix32(x);\n"
            "  float u = float(x >> 8) * (1.0f / 16777216.0f);\n\n"
            "The host ping-pongs (q_in, q_out) buffers across HMC steps; "
            "all dispatches share one command buffer for end-to-end timing."
        ),
        kernel_signatures=(
            "kernel void hmc_step(device const float *q_in        [[buffer(0)]],\n"
            "                     device       float *q_out       [[buffer(1)]],\n"
            "                     device       uint  *accept_cnt  [[buffer(2)]],\n"
            "                     device const float *A           [[buffer(3)]],\n"
            "                     constant uint  &K               [[buffer(4)]],\n"
            "                     constant uint  &d               [[buffer(5)]],\n"
            "                     constant uint  &L               [[buffer(6)]],\n"
            "                     constant float &eps             [[buffer(7)]],\n"
            "                     constant uint  &hmc_step_idx    [[buffer(8)]],\n"
            "                     constant uint  &seed            [[buffer(9)]],\n"
            "                     uint chain_idx [[thread_position_in_grid]]);\n"
            "\n"
            "Threads are dispatched 1-D, one per chain; guard with `if "
            "(chain_idx >= K) return;`. The host ping-pongs (q_in, q_out) "
            "between two K * d float buffers and increments hmc_step_idx "
            "by 1 per dispatch. accept_cnt is initialised to zero and "
            "accumulates accepted proposals over the run.\n"
            "\n"
            "All chosen sizes satisfy d <= 32 and d is even; thread-private "
            "arrays of size 32 are sufficient. Threadgroup-cooperative "
            "schemes (multiple threads per chain sharing the mat-vec) and "
            "simdgroup reductions are valid optimisations as long as the "
            "external buffer layout above is preserved."
        ),
        kernel_names=["hmc_step"],
        seed_path=_SEED,
        sizes=[
            # Three regimes chosen to span register pressure:
            #   d=8, K=16384  : light per-chain state, RNG/launch-overhead
            #                   matters more than mat-vec.
            #   d=16, K=4096  : balanced; per-thread arrays of 4*16 floats
            #                   fit comfortably in registers.
            #   d=32, K=1024  : per-thread state ~512 B; registers run hot,
            #                   register-spill or threadgroup-cooperative
            #                   schemes pay off.
            TaskSize("d8_K16384",  {"d": 8,  "K": 16384}),
            TaskSize("d16_K4096",  {"d": 16, "K":  4096}),
            TaskSize("d32_K1024",  {"d": 32, "K":  1024}),
        ],
        held_out_sizes=[
            TaskSize("d24_K2048",  {"d": 24, "K":  2048}),
        ],
    )

    def evaluate_size(self, harness, pipelines, size, chip, n_warmup, n_measure):
        d = int(size.params["d"])
        K = int(size.params["K"])
        L = _L
        eps = _EPS
        T = _T_HMC

        if d % 2 != 0:
            raise ValueError(f"d={d} must be even (Box-Muller pairs).")
        if d > 32:
            raise ValueError(f"d={d} > D_MAX=32 (seed kernel limit).")

        Sigma, A = _make_target(d)
        q0 = _make_init_q(K, Sigma)                    # (K, d) fp32

        # --- Buffers ----------------------------------------------------
        b_qA = harness.buf_from_np(q0)
        b_qB = harness.buf_zeros(q0.nbytes)
        b_acc = harness.buf_from_np(np.zeros(K, dtype=np.uint32))
        b_A = harness.buf_from_np(A)
        b_K = harness.buf_scalar(K, np.uint32)
        b_d = harness.buf_scalar(d, np.uint32)
        b_L = harness.buf_scalar(L, np.uint32)
        b_eps = harness.buf_scalar(eps, np.float32)
        b_seed = harness.buf_scalar(int(_RNG_SEED), np.uint32)
        # Pre-baked per-step indices; bind buffer 8 with offset = step * 4.
        step_indices = np.arange(T, dtype=np.uint32)
        b_step = harness.buf_from_np(step_indices)

        pso = pipelines["hmc_step"]
        max_tg = int(pso.maxTotalThreadsPerThreadgroup())
        tew = int(pso.threadExecutionWidth())
        tg_w = min(max_tg, max(tew, 64))
        grid_w = ((K + tg_w - 1) // tg_w) * tg_w

        view_qA = harness.np_view(b_qA, np.float32, K * d)
        view_qB = harness.np_view(b_qB, np.float32, K * d)
        view_acc = harness.np_view(b_acc, np.uint32, K)

        def reset():
            view_qA[:] = q0.ravel()
            view_qB[:] = 0.0
            view_acc[:] = 0

        def dispatch(enc):
            enc.setComputePipelineState_(pso)
            enc.setBuffer_offset_atIndex_(b_acc, 0, 2)
            enc.setBuffer_offset_atIndex_(b_A, 0, 3)
            enc.setBuffer_offset_atIndex_(b_K, 0, 4)
            enc.setBuffer_offset_atIndex_(b_d, 0, 5)
            enc.setBuffer_offset_atIndex_(b_L, 0, 6)
            enc.setBuffer_offset_atIndex_(b_eps, 0, 7)
            enc.setBuffer_offset_atIndex_(b_seed, 0, 9)
            for step in range(T):
                if step % 2 == 0:
                    qin, qout = b_qA, b_qB
                else:
                    qin, qout = b_qB, b_qA
                enc.setBuffer_offset_atIndex_(qin, 0, 0)
                enc.setBuffer_offset_atIndex_(qout, 0, 1)
                enc.setBuffer_offset_atIndex_(b_step, step * 4, 8)
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

        # --- Statistical correctness pass -------------------------------
        reset()
        harness.time_dispatch(dispatch)
        final_view = view_qA if T % 2 == 0 else view_qB
        q_final = final_view.copy().reshape(K, d).astype(np.float64)
        accept_total = int(view_acc.sum())
        accept_rate = accept_total / float(K * T)

        # Hard-fail any non-finite outputs (NaN/Inf usually mean a kernel
        # blew up — large eps, sign bug, energy divergence).
        if not np.all(np.isfinite(q_final)):
            n_bad = int(np.sum(~np.isfinite(q_final)))
            return SizeResult(
                size_label=size.label, correct=False,
                error_value=float(n_bad), error_kind="nonfinite_q",
                gpu_seconds=gpu_s, achieved=0.0, achieved_unit="GFLOPS",
                ceiling=float(chip.peak_fp32_gflops), ceiling_unit="GFLOPS",
                fraction_of_ceiling=0.0,
                extra={"d": d, "K": K, "L": L, "T": T, "eps": eps,
                       "accept_rate": accept_rate},
            )

        # Empirical mean and covariance over chains (treating each chain's
        # final state as one sample). With K >= 1024 IID samples and the
        # chain having mixed past q_0, the MC error in the mean is
        # ~ sqrt(diag(Sigma)) / sqrt(K).
        mean_emp = q_final.mean(axis=0)
        diff = q_final - mean_emp
        cov_emp = (diff.T @ diff) / float(K - 1)

        sigma_diag = np.sqrt(np.diag(Sigma).astype(np.float64))
        # Mean check: each component within tol_mean * sigma_i.
        mean_err_norm = float(np.max(np.abs(mean_emp) / sigma_diag))
        # Covariance check: relative Frobenius norm.
        Sigma64 = Sigma.astype(np.float64)
        cov_fro_err = float(np.linalg.norm(cov_emp - Sigma64, "fro")
                            / np.linalg.norm(Sigma64, "fro"))

        # Tolerances (see module docstring for the budget derivation):
        #   K=1024 IID samples → mean SE ~ 0.03 sigma per dim, so 0.15
        #   is ~5 sigma above noise. cov_fro_err ~ sqrt(2 d / K) for a
        #   correct sampler; 0.30 is generous for d=32, K=1024.
        tol_mean = 0.15
        tol_cov = 0.30
        # Accept window: lower bound catches "rejects everything"; upper
        # bound just below 1.0 catches "always accepts" while letting a
        # correct sampler with small eps land at ~0.99.
        acc_lo, acc_hi = 0.45, 0.999

        ok_mean = mean_err_norm <= tol_mean
        ok_cov = cov_fro_err <= tol_cov
        ok_acc = (acc_lo <= accept_rate <= acc_hi)
        correct = ok_mean and ok_cov and ok_acc

        # Combined error metric (worst of the three normalised checks)
        # for downstream readouts. The first two are already on a 0..1+
        # scale; we compute "how far past tolerance" for the accept rate
        # in the same scale.
        if accept_rate < acc_lo:
            acc_excess = (acc_lo - accept_rate) / acc_lo
        elif accept_rate > acc_hi:
            acc_excess = (accept_rate - acc_hi) / (1.0 - acc_hi + 1e-12)
        else:
            acc_excess = 0.0
        err = max(mean_err_norm / tol_mean,
                  cov_fro_err / tol_cov,
                  acc_excess)

        # --- Throughput metric ------------------------------------------
        # Useful FLOPs per HMC step per chain: (L + 1) force evaluations
        # at 2 d^2 - d FMAs each (a d x d mat-vec is d outputs of d-1
        # adds + d muls each ≈ 2 d^2 - d).
        flops_per_force = 2.0 * d * d - d
        flops_per_chain_per_step = (L + 1.0) * flops_per_force
        total_flops = K * T * flops_per_chain_per_step
        achieved = gflops(total_flops, gpu_s)
        ceiling = float(chip.peak_fp32_gflops)
        return SizeResult(
            size_label=size.label,
            correct=correct,
            error_value=err,
            error_kind="max(mean/tol, cov_fro/tol, accept_excess)",
            gpu_seconds=gpu_s,
            achieved=achieved,
            achieved_unit="GFLOPS (mat-vec FMAs only)",
            ceiling=ceiling,
            ceiling_unit="GFLOPS",
            fraction_of_ceiling=achieved / ceiling if ceiling > 0 else 0.0,
            extra={
                "d": d, "K": K, "L": L, "T": T, "eps": eps,
                "accept_rate": accept_rate,
                "mean_err_norm": mean_err_norm,
                "cov_fro_err": cov_fro_err,
                "tol_mean": tol_mean, "tol_cov": tol_cov,
                "acc_window": [acc_lo, acc_hi],
                "rng_seed": int(_RNG_SEED),
            },
        )
